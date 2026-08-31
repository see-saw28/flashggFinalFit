import law
import luigi
import os
import re
import subprocess
import ROOT
import uproot
import glob
import yaml
import errno
import shutil
import warnings
# Suppress // UserWarning: The value of the smallest subnormal for <class 'numpy.float64'> type is zero // warning.
warnings.filterwarnings("ignore", category=UserWarning, module="numpy.core.getlimits")
from scipy.stats import chi2

from commonTools import *
from commonObjects import *
# Helpers to extract/save pdfindex information for merged category flows
from pdfindex_utils import extract_pdf_indices, update_override_file

from Datacard.law_datacard import *
from Background.law_background import *

from framework import Task
from framework import HTCondorWorkflow, SlurmWorkflow

from pathlib import Path

# Ensure SLURM_JOB_ID is set for local workflows that still use /scratch.
if "SLURM_JOB_ID" not in os.environ:
    os.environ["SLURM_JOB_ID"] = f"local_{os.getpid()}"

# Function to safely create a directory
def safe_mkdir(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
        
def convert_boolean_string(string):
    if (string == "True") or (string == "true") or (string == True):
        return True
    else:
        return False

def get_lumi_label(year):
    if year in lumiMap:
        lumi = lumiMap[year]
    else:
        lumi = sum(lumiMap[y] for y in re.split(r"[,_]", str(year)) if y in lumiMap)
    return f"{lumi:.1f} fb^{{-1}} (13.6 TeV)"


_PDFINDEX_CACHE = {}

HIGGS_MASS = f"125.07"

def _save_specified_index_args(pdf_indices):
    save_specified_index = ",".join(pdf_indices)
    # Very large --saveSpecifiedIndex payloads can make combine v9.2.1
    # return success while producing tiny/zombie outputs. In that case,
    # skip saving the discrete pdfindex snapshot rather than poisoning law.
    if len(save_specified_index) > 8000:
        print(
            "Skipping --saveSpecifiedIndex because the argument is too long "
            f"({len(save_specified_index)} characters, {len(pdf_indices)} pdfindex categories)."
        )
        return []
    return ["--saveSpecifiedIndex", save_specified_index]

def _scan_parameter_range_args(config, poi):
    range_spec = config.get("combine_fit", {}).get("setParameterRange", "")
    if not range_spec:
        return []
    parts = [
        f"{poi}={part.split('=', 1)[1]}" if part.split("=", 1)[0] == "r" else part
        for part in range_spec.split(":")
    ]
    return ["--setParameterRanges", ":".join(parts)]

def execute_command(command, return_output=False, shell=False):
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True, shell=shell, env=os.environ)
        print("Script output:", result.stdout)
        print("Script executed successfully.")
        if return_output:
            return (result.stdout).split("\n")[0]
    except subprocess.CalledProcessError as e:
        print("Error executing script:", e.stderr)

def _parse_exclude_tokens(exclude_expr):
    """
    Parse exclude expressions into literal names + compiled regex patterns.

    Supported token forms (comma-separated):
    - literal parameter names
    - regex tokens in combineTool.py style: rgx{...}
    - regex tokens in /.../ style (convenience)
    """
    exclude_tokens = [t.strip() for t in (exclude_expr or "").split(",") if t.strip()]
    exclude_regex = []
    exclude_names = set()
    for tok in exclude_tokens:
        if tok.startswith("rgx{") and tok.endswith("}"):
            pattern = tok[4:-1]
            try:
                exclude_regex.append(re.compile(pattern))
            except Exception:
                exclude_names.add(tok)
            continue

        if len(tok) >= 2 and tok.startswith("/") and tok.endswith("/"):
            try:
                exclude_regex.append(re.compile(tok[1:-1]))
            except Exception:
                exclude_names.add(tok)
            continue

        exclude_names.add(tok)
    return exclude_names, exclude_regex

def _filter_names_by_exclude(names, exclude_expr):
    exclude_names, exclude_regex = _parse_exclude_tokens(exclude_expr)
    if not exclude_names and not exclude_regex:
        return list(names)

    kept = []
    for name in names:
        if name in exclude_names:
            continue
        if any(r.search(name) for r in exclude_regex):
            continue
        kept.append(name)
    return kept

def _list_modelconfig_nuisances(datacard_path, poi_list, exclude_expr=""):
    """
    Return nuisance parameter names from the workspace ModelConfig, excluding POIs and optional
    exclude patterns/names.
    """
    ws_file = ROOT.TFile.Open(datacard_path)
    if not ws_file or ws_file.IsZombie():
        raise RuntimeError(f"Could not open workspace file: {datacard_path}")
    w = ws_file.Get("w")
    if not w:
        raise RuntimeError(f"Workspace 'w' not found in: {datacard_path}")
    config = w.genobj("ModelConfig")
    if not config:
        raise RuntimeError(f"ModelConfig not found in workspace 'w' in: {datacard_path}")

    nuis = config.GetNuisanceParameters()
    if not nuis:
        return []

    names = []
    it = nuis.createIterator()
    var = it.Next()
    while var:
        name = var.GetName()
        if name not in poi_list and (not var.isConstant()) and var.InheritsFrom("RooRealVar"):
            names.append(name)
        var = it.Next()

    names = _filter_names_by_exclude(names, exclude_expr)
    names.sort()
    return names
        
def manually_copy_t3(src, dst):
    # List files in the directory
    list_command = ["xrdfs", "root://t3dcachedb03.psi.ch", "ls", src]
    file_list = subprocess.check_output(list_command).decode().splitlines()
    
    print(file_list)

    # Copy each file
    for file_path in file_list:
        filename = file_path.split("/")[-1]
        if "bkgfTest-Data" in filename: continue
        if "/pnfs" in file_path: 
            src_file = f"root://t3dcachedb03.psi.ch:1094/{file_path}"
            dest_file = f"root://t3dcachedb03.psi.ch:1094/{dst}/{filename}"
            
            print(f"Copying {filename}...")
            print(f"xrdcp -rf {src_file} {dest_file}")
            execute_command([f"xrdcp -rf {src_file} {dest_file}"], shell=True)
        else:
            print(f"Copying {filename}...")
            print(f"cp -rf {file_path} {dst}/{filename}")
            execute_command([f"cp -rf {file_path} {dst}/{filename}"], shell=True)

def manually_move_t3(src, dst):
    # List files in the directory
    list_command = ["xrdfs", "root://t3dcachedb03.psi.ch", "ls", src]
    file_list = subprocess.check_output(list_command).decode().splitlines()
    
    print(file_list)

    # Copy each file
    for file_path in file_list:
        filename = file_path.split("/")[-1]
        if "bkgfTest-Data" in filename: continue
        if "/pnfs" in file_path: 
            src_file = f"root://t3dcachedb03.psi.ch:1094/{file_path}"
            dest_file = f"root://t3dcachedb03.psi.ch:1094/{dst}/{filename}"
            
            print(f"Moving {filename} on or off the PSI SE...")
            
            print(f"xrdfs root://t3dcachedb03.psi.ch:1094/ mv -f {src_file} {dest_file}")
            execute_command([f"xrdfs root://t3dcachedb03.psi.ch:1094/ mv -f {src_file} {dest_file}"], shell=True)
        else:
            print(f"Moving {filename}...")
            print(f"mv -f {file_path} {dst}/{filename}")
            execute_command([f"mv -f {file_path} {dst}/{filename}"], shell=True)
    
class PrepareTheDirectory(MultiYearTask):#(law.Task): #(Task, HTCondorWorkflow, law.LocalWorkflow):

    # def requires(self):
    def _requires_single(self):
        
        tasks = {}
        # Load the input configuration
        config = self.get_input_config()
        
        output_dir = self.get_output_dir()
        
        yieldsConfig = config['datacard_yields']
        
            
        tasks["MakeDatacard"] = MakeDatacard.req(
            self,
            output_dir=output_dir, 
            workflow=yieldsConfig["execution"], 
            slurm_partition=yieldsConfig['batchPartition'], 
            slurm_memory=yieldsConfig['batchMemory'], 
            slurm_max_runtime=yieldsConfig['batchMaxRuntime'], 
            htcondor_partition=yieldsConfig['batchPartition'], 
            htcondor_memory=yieldsConfig['batchMemory'], 
            htcondor_max_runtime=yieldsConfig['batchMaxRuntime'])

        # tasks["Background"] = Background.req(
        #     self,
        #     output_dir=output_dir
        # )

        return tasks
    
    def create_branch_map(self):
        # map branch indexes to ascii numbers from 97 to 122 ("a" to "z")    
        branch_map = {i: i for i in range(1)}
        return branch_map

    def output(self):
        outdir = f"outdir_{self.year}_{self.variable}"
                
        # Load the input configuration
        config = self.get_input_config()
        output_dir = self.get_output_dir()

        if self.variable == '':
            fitFolderName = f'runFits_mu_fiducial'
        else:
            fitFolderName = f'runFits_{self.variable}'

        input_path = config['inputFiles']['Trees2WSData']

        bkgConfig = config["backgroundScriptCfg"]
        if bkgConfig['cats'] == 'auto':
            bkgConfig['cats'] = (extractListOfCatsFromHiggsDNAAllData(input_path))

        packagedConfig = config[f"packaged_{self.year}"]
        outputExt = packagedConfig['ext']

        cat_list = bkgConfig['cats'].split(",")

        signal_model_folder_name = config['datacard_yields']['sigModelWSDir'].split('/')[-2]
        background_model_folder_name = config['datacard_yields']['bkgModelWSDir'].split('/')[-2]

        output_data = []

        output_data.append(os.path.join(output_dir, 'Combine', outdir))

        output_data.append(os.path.join(output_dir, 'Combine', outdir, fitFolderName))

        years = yearMap[self.year]
        if len(years) > 1:
            raise RuntimeError(f"PrepareTheDirectory expects a single year, but got: {years}. Run CombineDatacards instead.")


        if signal_model_folder_name == background_model_folder_name:
            model_folder_name = signal_model_folder_name
            output_data.append(os.path.join(output_dir, 'Combine', outdir, model_folder_name))
            output_data.append(os.path.join(output_dir, 'Combine', outdir, model_folder_name, 'background'))
            output_data.append(os.path.join(output_dir, 'Combine', outdir, model_folder_name, 'signal'))
        else:
            output_data.append(os.path.join(output_dir, 'Combine', outdir, signal_model_folder_name))
            output_data.append(os.path.join(output_dir, 'Combine', outdir, signal_model_folder_name, 'signal'))

            output_data.append(os.path.join(output_dir, 'Combine', outdir, background_model_folder_name))
            output_data.append(os.path.join(output_dir, 'Combine', outdir, background_model_folder_name, 'background'))

        for cat in cat_list:
            output_data.append(os.path.join(output_dir, 'Combine', outdir, background_model_folder_name, 'background', f'CMS-HGG_multipdf_{cat}.root'))
            output_data.append(os.path.join(output_dir, 'Combine', outdir, signal_model_folder_name, 'signal', f'CMS-HGG_sigfit_packaged{outputExt}_{cat}.root'))


        # Define the file paths
        if self.variable == '':
            output_data.append(os.path.join(output_dir, 'Combine', outdir, f'Datacard_{self.year}.txt'))
        else:
            output_data.append(os.path.join(output_dir, 'Combine', outdir, f'Datacard_{self.variable}_{self.year}.txt'))

        for i, output in enumerate(output_data):
            output_data[i] = law.LocalFileTarget(output)

        return output_data

    def run(self):
        background_suffix = f""
        outdir = f"outdir_{self.year}_{self.variable}"

        if self.variable == '':
            fitFolderName = f'runFits_mu_fiducial'
        else:
            fitFolderName = f'runFits_{self.variable}'

        #Load central config file
        config = self.get_input_config()
        output_dir = self.get_output_dir()

        # Creating the Combine directory alongside the Models dir
        if self.batch_flavor == "slurm/psi":
            execute_command([f"xrdfs root://t3dcachedb03.psi.ch:1094/ mkdir -p {os.path.join(output_dir, 'Combine', outdir, fitFolderName)}"], shell=True)
        else:
            safe_mkdir(os.path.join(output_dir, 'Combine', outdir))
            safe_mkdir(os.path.join(output_dir, 'Combine', outdir))
            safe_mkdir(os.path.join(output_dir, 'Combine', outdir, fitFolderName))

        signal_model_folder_name = config['datacard_yields']['sigModelWSDir'].split('/')[-2]
        background_model_folder_name = config['datacard_yields']['bkgModelWSDir'].split('/')[-2]

        years = yearMap[self.year]
        if len(years) > 1:
            raise RuntimeError(f"PrepareTheDirectory expects a single year, but got: {years}. Run CombineDatacards instead.")
        
    
        if signal_model_folder_name == background_model_folder_name:
            Model_dst_path = os.path.join(output_dir, 'Combine', outdir, signal_model_folder_name)
            background_dst_path = os.path.join(output_dir, 'Combine', outdir, signal_model_folder_name, 'background'+background_suffix)
            signal_dst_path = os.path.join(output_dir, 'Combine', outdir,signal_model_folder_name, 'signal')
            if self.batch_flavor == "slurm/psi":
                execute_command([f'xrdfs root://t3dcachedb03.psi.ch:1094/ mkdir -p {Model_dst_path}'], shell=True)
            else:
                safe_mkdir(Model_dst_path)
        else:
            signalModel_dst_path = os.path.join(output_dir, 'Combine', outdir, signal_model_folder_name)
            signal_dst_path = os.path.join(output_dir, 'Combine', outdir, signal_model_folder_name, 'signal')
            if self.batch_flavor == "slurm/psi":
                execute_command([f'xrdfs root://t3dcachedb03.psi.ch:1094/ mkdir -p {signalModel_dst_path}'], shell=True)
            else:
                safe_mkdir(signalModel_dst_path)
            
            backgroundModel_dst_path = os.path.join(output_dir, 'Combine', outdir, background_model_folder_name)
            background_dst_path = os.path.join(output_dir, 'Combine', outdir, background_model_folder_name, 'background'+background_suffix)
            if self.batch_flavor == "slurm/psi":
                execute_command([f'xrdfs root://t3dcachedb03.psi.ch:1094/ mkdir -p {backgroundModel_dst_path}'], shell=True)
            else:
                safe_mkdir(backgroundModel_dst_path)

        if self.batch_flavor == "slurm/psi":
            execute_command([f'xrdfs root://t3dcachedb03.psi.ch:1094/ mkdir -p {signal_dst_path}'], shell=True)
            execute_command([f'xrdfs root://t3dcachedb03.psi.ch:1094/ mkdir -p {background_dst_path}'], shell=True)
        else:
            safe_mkdir(signal_dst_path)
            safe_mkdir(background_dst_path)

        # Copying relevant files in Models directory
        background_src_path = os.path.join(output_dir, "Background", f"outdir_{config['backgroundScriptCfg']['ext']}"+background_suffix)
        signal_src_path = os.path.join(output_dir, 'Signal', f"outdir_packaged{config[f'packaged_{self.year}']['ext']}_{self.year}/")

        if self.batch_flavor == "slurm/psi":
            manually_copy_t3(background_src_path, background_dst_path)
            manually_copy_t3(signal_src_path, signal_dst_path)
        else:
            shutil.copytree(background_src_path, background_dst_path, dirs_exist_ok=True)
            shutil.copytree(signal_src_path, signal_dst_path, dirs_exist_ok=True)

        # IDK for what that is useful
        path_pattern = f"{signal_model_folder_name}/signal/*_{self.year}.root"

        # Use glob to find all matching files
        for file_path in glob.glob(path_pattern):
            if os.path.isfile(file_path):  # Check if it's a file
                # Remove "_{self.year}" from the filename
                new_name = file_path.replace(f"_{self.year}.root", ".root")
                
                # Rename the file
                os.rename(file_path, new_name)
                
                print(f"Renamed {file_path} to {new_name}")


        # Define the file paths
        if self.variable == '':
            datacard_file_cleaned = os.path.join(output_dir, 'Datacards', f'Datacard_{self.year}_cleaned.txt')
            datacard_file = os.path.join(output_dir, 'Datacards', f'Datacard_{self.year}.txt')
            destination_file = os.path.join(output_dir, 'Combine', outdir, f'Datacard_{self.year}.txt')
        else:
            datacard_file_cleaned = os.path.join(output_dir, 'Datacards', f'Datacard_{self.variable}_{self.year}_cleaned.txt')
            datacard_file = os.path.join(output_dir, 'Datacards', f'Datacard_{self.variable}_{self.year}.txt')
            destination_file = os.path.join(output_dir, 'Combine', outdir, f'Datacard_{self.variable}_{self.year}.txt')

        # Check if the cleaned file exists
        if os.path.exists(datacard_file_cleaned):
            # Copy the cleaned file if it exists
            if self.batch_flavor == "slurm/psi":
                execute_command([f'xrdcp -rf root://t3dcachedb03.psi.ch:1094/{datacard_file_cleaned} root://t3dcachedb03.psi.ch:1094/{destination_file}'], shell=True)
            else:
                shutil.copy2(datacard_file_cleaned, destination_file)
        else:
            # Otherwise, copy the uncleaned file
            if self.batch_flavor == "slurm/psi":
                execute_command([f'xrdcp -rf root://t3dcachedb03.psi.ch:1094/{datacard_file} root://t3dcachedb03.psi.ch:1094/{destination_file}'], shell=True)
            else:
                shutil.copy2(datacard_file, destination_file)
            


        print("Combine directory sucessfully prepared.")

def is_signal_only_datacard(config):
    return convert_boolean_string(config.get("datacard_yields", {}).get("skipBkg", False))


class CombineDatacards(Task):

    def requires(self):
        years = yearMap[self.year]
        if len(years) < 2:
            raise RuntimeError(f"CombineDatacards requires at least two years to combine, but got: {years}")
        
        return {y: PrepareTheDirectory.req(self,years=y) for y in years}
        

    def run(self):
        # --- paths relative to 'law' ---

        output_dir = self.get_output_dir()

        outdir = f"outdir_{self.year}_{self.variable}" if self.variable != '' else f"outdir_{self.year}"

        combined_label = self.year
        
        years = yearMap[self.year]

        combined_output_dir = Path(output_dir) / "Combine" / outdir

        
        combined_output_dir.mkdir(parents=True, exist_ok=True)

        

        # --- 1 and 2. Copy Datacards and Get output paths from configs --
        year_output_paths = {}
        for y in years:
            src = Path(self.input()[y][-1].path)
            year_output_paths[y] = src.parent 
            dst = combined_output_dir / f"Datacard_{self.variable}_{y}.txt"
            if src.exists():
                shutil.copy2(src, dst)
                print(f"Copied datacard for {y}")
            else:
                print(f"⚠️  Missing datacard: {src}")

        # --- 3. Copy Models ---
        if self.variable in ['', 'MH']:
            for y, out_path in year_output_paths.items():
                src_models = out_path / "Models"
                dst_models = combined_output_dir / f"Models_{y}"
                if dst_models.exists():
                    print(f"Models for {y} already exist in combined output directory: {dst_models}")
                elif src_models.exists():
                    shutil.copytree(src_models, dst_models, dirs_exist_ok=True)
                    print(f"Copied models for {y}")
                else:
                    print(f"Missing models directory: {src_models}")
        else:
            for y, out_path in year_output_paths.items():
                src_models = out_path / f"Models_{self.variable}"
                dst_models = combined_output_dir / f"Models_{self.variable}_{y}"
                if src_models.exists():
                    shutil.copytree(src_models, dst_models, dirs_exist_ok=True)
                    print(f"Copied models for {y}")
                else:
                    print(f"Missing models directory: {src_models}")

        # --- 4. Combine datacards ---
        os.chdir(combined_output_dir)
        print(f"Changed working directory to {combined_output_dir}")
        if self.variable == '':
            card_args = " ".join([f"Y{y[-2:]}=Datacard_{y}.txt" for y in years])
            combined_card = f"Datacard_{combined_label}.txt"
        else:
            card_args = " ".join([f"Y{y[-2:]}=Datacard_{self.variable}_{y}.txt" for y in years])
            combined_card = f"Datacard_{self.variable}_{combined_label}.txt"

        print("Combining datacards...")
        cmd = f"combineCards.py {card_args} > {combined_card}"
        print(f"Running command: {cmd}")
        subprocess.run(cmd, shell=True, check=True)

        # --- 5. Fix model paths per year ---
        print("Fixing model paths per year...")
        if self.variable in ['', 'MH']:
            for y in years:
                tag = f"Y{y[-2:]}"
                models_dir = f"./Models_{y}/"
                subprocess.run(
                    f"sed -i -E '/^shapes\\s+\\S+\\s+{tag}_/ s|(\\s)\\./Models/|\\1{models_dir}|g' {combined_card}",
                    shell=True,
                    check=True,
                )
        else:
            for y in years:
                tag = f"Y{y[-2:]}"
                models_dir = f"./Models_{self.variable}_{y}/"
                subprocess.run(
                    f"sed -i -E '/^shapes\\s+\\S+\\s+{tag}_/ s|(\\s)\\./Models_{self.variable}/|\\1{models_dir}|g' {combined_card}",
                    shell=True,
                    check=True,
                )

        # --- 6. Delete some stuff ---
        if self.variable == '':
            for y in years:
                tmp_path = combined_output_dir / f"Datacard_{y}.txt"
                if tmp_path.exists():
                    tmp_path.unlink()
        else:
            for y in years:
                tmp_path = combined_output_dir / f"Datacard_{self.variable}_{y}.txt"
                if tmp_path.exists():
                    tmp_path.unlink()

        print(f"Combined workspace created in {combined_output_dir}")
    
    def output(self):
        years = yearMap[self.year]
        if len(years) < 2:
            # No combined output for single year
            return None

        # Use the repository root (same base_dir as in run()) instead of cwd.
        base_dir = Path(__file__).resolve().parent.parent
        combined_label = self.year
        outdir = f"outdir_{self.year}_{self.variable}" if self.variable != '' else f"outdir_{self.year}"

        if self.variable == '':
            combined_card_path = os.path.join(
                base_dir,
                f"output_{combined_label}_inclusive/Combine/{outdir}/Datacard_{combined_label}.txt"
            )
        else:
            combined_card_path = os.path.join(
                base_dir,
                f"output_{combined_label}_{self.variable}/v{self.version}/Combine/{outdir}/Datacard_{self.variable}_{combined_label}.txt"
            )
        return law.LocalFileTarget(combined_card_path)
    



class RunText2Workspace(MultiYearTask): #(law.Task): #(Task, HTCondorWorkflow, law.LocalWorkflow):
    
    # def requires(self):
    def _requires_single(self):
        
        output_dir = self.get_output_dir()

        tasks = {}

        years = yearMap[self.year]
        
        if len(years) > 1:
            tasks["CombineDatacards"] = CombineDatacards.req(self,output_dir=output_dir)
        else:
            tasks["PrepareTheDirectory"] = PrepareTheDirectory.req(self,output_dir=output_dir)
        
        return tasks


    def output(self):
        # Load the input configuration
        config = self.get_input_config()
        output_dir = self.get_output_dir()

        outdir = f"outdir_{self.year}_{self.variable}" if self.variable != '' else f"outdir_{self.year}"

        # Define the file paths
        if self.variable == '':
            output = [os.path.join(output_dir, 'Combine', outdir, f'Datacard_{self.year}.root')]
        else:
            output = [os.path.join(output_dir, 'Combine', outdir, f'Datacard_{self.variable}_{self.year}.root')]

        outputFileTargets = []

        for _, current_output_path in enumerate(output):
            outputFileTargets.append(law.LocalFileTarget(current_output_path))
        
        # print(outputFileTargets)

        return outputFileTargets

    def run(self):      

        if self.variable == "":
            mode = "mu_fiducial"
            datacard_name = f"Datacard_{self.year}"
            
        elif self.variable == "tuto":
            mode = "mu_fiducial"
            datacard_name = f"Datacard_{self.variable}_{self.year}"
        else:
            mode = self.variable
            datacard_name = f"Datacard_{self.variable}_{self.year}"

        workspace_name = datacard_name

        config = self.get_input_config()
        output_dir = self.get_output_dir()
            
        script_path = os.path.join(os.environ["ANALYSIS_PATH"],"Combine/RunText2Workspace.py")
        
        if self.batch_flavor == "slurm/psi":
            # Have to use /scratch/batch_username/ for slurm/psi
            os.environ["TARGET_PATH"] = f"/scratch/{os.environ['USER']}/{os.environ['SLURM_JOB_ID']}"
            execute_command([f'mkdir -p $TARGET_PATH/Combine/{config["datacard_yields"]["sigModelWSDir"]}'], shell=True)
            execute_command([f'mkdir -p $TARGET_PATH/Combine/{config["datacard_yields"]["bkgModelWSDir"]}'], shell=True)
            # Keep t2w_jobs for debugging purposes
            execute_command([f'mkdir -p $TARGET_PATH/Combine/outdir_{self.year}_{self.variable}/t2w_jobs'], shell=True)    
            temp_output_dir = os.environ["TARGET_PATH"]
            
            # Copy concerning datacard + Model to scratch dir, cause of how RunText2Workspace works..
            # Copying datacard...
            slurm_copy_command = [
                'xrdcp', '-rf',
                'root://t3dcachedb03.psi.ch:1094//'+ f'{output_dir}/Combine/{datacard_name}.txt',
                f'{temp_output_dir}/Combine'
            ]
            execute_command(slurm_copy_command)
            # Copying Signal Model...
            slurm_copy_command = [
                'xrdcp', '-rf',
                'root://t3dcachedb03.psi.ch:1094//'+ f'{output_dir}/Combine/{config["datacard_yields"]["sigModelWSDir"]}',
                f'{temp_output_dir}/Combine/{config["datacard_yields"]["sigModelWSDir"].split("/")[-2]}'
            ]
            execute_command(slurm_copy_command)
            # Copying Background Model...
            slurm_copy_command = [
                'xrdcp', '-rf',
                'root://t3dcachedb03.psi.ch:1094//'+ f'{output_dir}/Combine/{config["datacard_yields"]["bkgModelWSDir"]}',
                f'{temp_output_dir}/Combine/{config["datacard_yields"]["bkgModelWSDir"].split("/")[-2]}'
            ]
            execute_command(slurm_copy_command)
        else:
            temp_output_dir = output_dir

        outdir = f"outdir_{self.year}_{self.variable}" if self.variable != '' else f"outdir_{self.year}"
        datacards_dir = os.path.join(temp_output_dir, 'Combine', outdir)

        arguments = [
            "python3",
            script_path,
            "--inputName", datacard_name,
            "--outputDir", datacards_dir,
            "--outputName", workspace_name,
            "--mode", mode,
            "--common_opts", f"-m {HIGGS_MASS} higgsMassRange=122,128 --channel-masks",
            "--batch", "local",
        ]
        if self.variable != '':
            arguments.append("--ext")
            arguments.append(f"{self.variable}")
        command = arguments
        print("RunText2Workspace: " + " ".join(command))
        try:
            result = subprocess.run(command, check=True, text=True, capture_output=True)
            print("Script output:", result.stdout)
            print("Script executed successfully.")
        except subprocess.CalledProcessError as e:
            print("Error executing script:", e.stderr)
        
        if self.batch_flavor == "slurm/psi":
            # Copy workspaces to workspaces folder
            list_command = ["ls", os.path.join(temp_output_dir, 'Combine', outdir, 't2w_jobs')]
            file_list = subprocess.check_output(list_command).decode().splitlines()

            print(file_list)
            execute_command([f'xrdcp -rf {datacards_dir}/{datacard_name}.root root://t3dcachedb03.psi.ch:1094//{output_dir}/Combine/'], shell=True)
            execute_command([f"xrdcp -rf {os.path.join(temp_output_dir, 'Combine', outdir, 't2w_jobs/')} root://t3dcachedb03.psi.ch:1094//{output_dir}/Combine/t2w_jobs/"], shell=True)
            shutil.rmtree(temp_output_dir)

        # Persist the pdfindex values from the produced workspace so downstream
        # cat-merged fits can reuse the same indices without recomputing them.
        final_root_path = os.path.join(output_dir, 'Combine', outdir, f'{datacard_name}.root')
        pdf_indices = extract_pdf_indices(final_root_path)
        if pdf_indices:
            override_path = os.path.join(os.environ["ANALYSIS_PATH"], "config", "pdfindex_overrides.json")
            variable_key = self.variable if self.variable != '' else 'inclusive'
            update_override_file(override_path, self.year, variable_key, pdf_indices)
        
        
class AsimovFitCategoryFirstStep(Task, law.LocalWorkflow, HTCondorWorkflow, SlurmWorkflow): #(law.Task): #(Task, HTCondorWorkflow, law.LocalWorkflow):
    output_dir = law.Parameter(default = '', description="Path to the output directory")
    variable = law.Parameter(default="", description="Variable to be used")
    year = law.Parameter(default='2022', description="Year")
    cats = law.Parameter(description="Current category")
    
    batch_flavor = law.Parameter(default="htcondor", description="Batch system to use")

    # def requires(self):
    def workflow_requires(self):
        workflow_reqs = super().workflow_requires()

        tasks = {}

        if workflow_reqs:
            tasks.update(workflow_reqs)
        
        output_dir = self.get_output_dir()
    
  
        tasks["RunT2WS"] = RunText2Workspace.req(
            self,
            output_dir=output_dir, 
            years=self.year,
            )        

        return tasks

    def create_branch_map(self):

        if self.variable == '':
            branch_map = {i: cat for i, cat in enumerate(["r"])}
        else:
            nCats = len(self.cats.split(","))
            
            cat_list = [
                self.cats.split(",")[categoryIndex]
                for categoryIndex in range(nCats)
            ]
            
            branch_map = {i: cat for i, cat in enumerate(cat_list)}
        return branch_map

    def output(self):
        current_branch = self.branch_data
        
        output_dir = self.get_output_dir()

        if self.variable == '':
            fitFolderName = f'runFits_mu_fiducial'
        else:
            fitFolderName = f'runFits_{self.variable}'

        outdir = f"outdir_{self.year}_{self.variable}" if self.variable != '' else f"outdir_{self.year}"
            
        output = [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'asimov')]

        output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'asimov', f'higgsCombinefirstStep_{current_branch}.MultiDimFit.mH{HIGGS_MASS}.root')]
        output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'asimov', f'multidimfitfirstStep_{current_branch}.root')]
        
        outputFileTargets = []
                
        for _, current_output_path in enumerate(output):
            outputFileTargets.append(law.LocalFileTarget(current_output_path))

        return outputFileTargets

    def run(self):
        current_branch = self.branch_data

        output_dir = self.get_output_dir()

        outdir = f"outdir_{self.year}_{self.variable}" if self.variable != '' else f"outdir_{self.year}"

        if self.variable == '':
            fitFolderName = f'runFits_mu_fiducial'
        else:
            fitFolderName = f'runFits_{self.variable}'  
            
        if self.variable == '':
            datacard_path = os.path.join(output_dir, 'Combine', outdir, f'Datacard_{self.year}.root')
        else:
            datacard_path = os.path.join(output_dir, 'Combine', outdir, f'Datacard_{self.variable}_{self.year}.root')

        cwd = os.getcwd()

        if self.batch_flavor == "slurm/psi":
            # Have to use /scratch/batch_username/ for slurm/psi
            if "/work" in output_dir:
                execute_command([f'mkdir -p {output_dir}/Combine/{fitFolderName}/asimov'], shell=True)
            else:   
                execute_command([f'xrdfs root://t3dcachedb03.psi.ch:1094/ mkdir -p {output_dir}/Combine/{fitFolderName}/asimov'], shell=True)

            os.environ["TARGET_PATH"] = f"/scratch/{os.environ['USER']}/{os.environ['SLURM_JOB_ID']}"
            execute_command([f'mkdir -p $TARGET_PATH/Combine/{fitFolderName}/asimov'], shell=True)
            os.chdir(os.path.join(os.environ["TARGET_PATH"], 'Combine', fitFolderName, 'asimov'))
        else:
            execute_command([f'mkdir -p {output_dir}/Combine/{outdir}/{fitFolderName}/asimov'], shell=True)
            os.chdir(os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'asimov'))

        # Split the year string into a list
        years = self.year.split("_")

        if self.variable in ["", "tuto"]:
            # Make all combinations of BMW and years: This also works if self.year is 2022_2023 in a combineCards workflow!
            pdf_indices = [f"pdfindex_{bmw}_{year}_13TeV" for bmw in BMW for year in years]

            arguments = [
                "combine",
                "-M", "MultiDimFit",
                datacard_path,
                "--freezeParameters", "MH",
                "-m", f"{HIGGS_MASS}",
                "-n", f"firstStep_{current_branch}",
                "--cminDefaultMinimizerStrategy=0",
                "--cminFallbackAlgo", "Minuit2,Migrad,1:10",
                "--expectSignal", "1",
                "--saveWorkspace",
                "--X-rtd", "MINIMIZER_freezeDisassociatedParams",
                "--X-rtd", "MINIMIZER_multiMin_hideConstants",
                "--X-rtd", "MINIMIZER_multiMin_maskConstraints",
                "--X-rtd", "MINIMIZER_multiMin_maskChannels=2",
                "-t", "-1",
                "--saveFitResult", #pdfindex_cat0_{self.year}_13TeV,pdfindex_cat1_2022_13TeV,pdfindex_cat2_2022_13TeV}
                "--floatOtherPOIs", "1"
            ]
            arguments.extend(_save_specified_index_args(pdf_indices))
            command = arguments
            
            
        elif self.variable != '':
            if self.variable != "MH":
                pdf_indices = combineVariableDict(self.variable, self.year)['pdfIndeces']
                cache_key = (datacard_path,)
                if cache_key not in _PDFINDEX_CACHE and os.path.exists(datacard_path):
                    datacard_pdf_indices = extract_pdf_indices(datacard_path)
                    if datacard_pdf_indices:
                        _PDFINDEX_CACHE[cache_key] = datacard_pdf_indices
                if cache_key in _PDFINDEX_CACHE:
                    pdf_indices = _PDFINDEX_CACHE[cache_key]
            else:
                pdf_indices = None
                
            arguments = [
                "combine",
                "-M", "MultiDimFit",
                datacard_path,
                # FIXME
                "--freezeParameters", "r",
                "-m", f"{HIGGS_MASS}",
                "-n", f"firstStep_{current_branch}",
                "--setParameters", "r=1.0",
                "--cminDefaultMinimizerStrategy=0",
                "--cminFallbackAlgo", "Minuit2,Migrad,1:10",
                "--saveWorkspace",
                "--X-rtd", "MINIMIZER_freezeDisassociatedParams",
                "--X-rtd", "MINIMIZER_multiMin_hideConstants",
                "--X-rtd", "MINIMIZER_multiMin_maskConstraints",
                "--X-rtd", "MINIMIZER_multiMin_maskChannels=2",
                "-t", "-1",
                "-P", f"{current_branch}",
                "--saveFitResult",
                "--floatOtherPOIs", "1"
            ]
            if self.variable != 'MH':
                arguments.extend(_save_specified_index_args(pdf_indices))
                arguments.append("--setParameters")
                arguments.append(f"""{",".join(combineVariableDict(self.variable, self.year)['paramStr'])}""")
            # else: 
            #     arguments.append()

        command = arguments
        print(' '.join(command))
        try:
            result = subprocess.run(command, check=True, text=True, capture_output=True)
            print("Script output:", result.stdout)
            print("Script executed successfully.")
        except subprocess.CalledProcessError as e:
            print("Error executing script:", e.stderr)
            raise
        
        # Copy the files back to pnfs if we are on slurm/psi
        if self.batch_flavor == "slurm/psi":
            # Have to copy over the output to the final directory
            # Don't forget to VOMS!
            if "/work" in output_dir:
                slurm_copy_command = [
                    'cp', '-rf',
                    f"{os.environ['TARGET_PATH']}/Combine/",
                    output_dir
                ]
            else:
                slurm_copy_command = [
                    'xrdcp', '-rf',
                    f"{os.environ['TARGET_PATH']}/Combine/",
                    'root://t3dcachedb03.psi.ch:1094//'+output_dir
                ]
            print(slurm_copy_command)
            execute_command(slurm_copy_command)
            # Clean up the temporary directory
            shutil.rmtree(os.environ["TARGET_PATH"])
            
        os.chdir(cwd)
        
class CreateAsimovFitFirstStep(Task): #(law.Task): #(Task, HTCondorWorkflow, law.LocalWorkflow):
        
    def requires(self):
        
        config = self.get_input_config()
        output_dir = self.get_output_dir()
            
        tasks = []
        if self.variable in ["", "tuto"]:
            cats = "r"
        elif self.variable == "MH":
            cats = "MH"
        else:
            cats = ",".join(combineVariableDict(self.variable, self.year)['paramStrNoOne'])
        
        impactConfig = config['combine_impacts']
        
        tasks += [AsimovFitCategoryFirstStep.req(self, output_dir=output_dir, cats=cats, workflow=impactConfig["execution"], slurm_partition=impactConfig['batchPartition'], slurm_memory=impactConfig['batchMemory'], slurm_max_runtime=impactConfig['batchMaxRuntime'], htcondor_partition=impactConfig['batchPartition'], htcondor_memory=impactConfig['batchMemory'], htcondor_max_runtime=impactConfig['batchMaxRuntime'])]
            
        return tasks
    
    def create_branch_map(self):
        branch_list = [0]
        
        branch_map = {i: branch for i, branch in enumerate(branch_list)}
        return branch_map

    def output(self):
        return self.input()

    def run(self):
        return True

# Handles both standard per-category scans and the merged-category flow by
# optionally branching over a comma-separated list of categories. When
# `cats` is set all bins are executed inside one HTCondor job to avoid
# spawning one submission per differential bin.
class AsimovFitCategorySyst(Task, law.LocalWorkflow, HTCondorWorkflow, SlurmWorkflow): #(law.Task): #(Task, HTCondorWorkflow, law.LocalWorkflow):
    cat = law.Parameter(default="", description="Current category")
    cats = law.Parameter(default="", description="Comma separated list of categories to process")
    nPoints = law.Parameter(default=30, description="Number of points for the LL scan")
    set_pdfidx_inclusives = law.Parameter(default=False, description="Year") # convert_boolean_string
    freeze = law.Parameter(default="", description="Parameters to freeze in the fit")

    # def requires(self):
    def workflow_requires(self):
        workflow_reqs = super().workflow_requires()

        tasks = {}

        if workflow_reqs:
            tasks.update(workflow_reqs)
        
        output_dir = self.get_output_dir()
               
        tasks["CreateAsimovFitFirstStep"] = CreateAsimovFitFirstStep.req(self, output_dir=output_dir)
        
        return tasks
    
    def create_branch_map(self):
        points = list(range(int(self.nPoints)))
        if not self.cats:
            return {i: point for i, point in enumerate(points)}

        # Differential fits with cat-merging provide a csv list of categories,
        # here we build the cartesian product (cat, scan point) so every branch
        # can be processed inside the same Condor submission.
        cat_list = [self.cats]
        branch_map = {}
        idx = 0
        for cat in cat_list:
            for point in points:
                branch_map[idx] = (cat, point)
                idx += 1
        return branch_map

    def _current_branch_info(self):
        # Branch data can be just the point index (legacy behaviour) or a
        # (category, point) tuple when running in cat-merged mode, which lets
        # all bins share one scheduler job.
        data = self.branch_data
        if isinstance(data, tuple):
            return data
        if not self.cat:
            raise ValueError("Category not set for branch without tuple data")
        return (self.cat, data)

    def output(self):
        current_cat, current_point = self._current_branch_info()
        current_dir = "_".join(current_cat.split(",")) if current_cat != self.cat else ""
        
        output_dir = self.get_output_dir()

        if self.variable == '':
            fitFolderName = f'runFits_mu_fiducial'
        else:
            fitFolderName = f'runFits_{self.variable}'

        outdir = f'outdir_{self.year}_{self.variable}' if self.variable != '' else '' 
            
        output = [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'asimov')]

        output = [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'asimov', current_dir, self.freeze)]
        
        output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'asimov', current_dir, self.freeze, f'higgsCombineAsimovPostFitScanFit_{self.cat}.POINTS.{current_point}.{current_point}.MultiDimFit.mH{HIGGS_MASS}.root')]
        
        outputFileTargets = []
                
        for _, current_output_path in enumerate(output):
            outputFileTargets.append(law.LocalFileTarget(current_output_path))
            
        return outputFileTargets
    
    def make_channel_mask_params(self, cats):
        years = yearMap[self.year]

        cats_analysis = self.get_cats()
        cats_analysis = [cat.strip() for cat in cats_analysis.split(",") if cat.strip()]

        # Single-year case: channels are CAT0, CAT1, ...
        if len(years) == 1:
            return ",".join(
                f"mask_{ch}={0 if ch in cats.split(',') else 1}"
                for ch in cats_analysis
            )

        # Multi-year case: channels are Y22_CAT0, Y23_CAT0, ...
        mask_params = []
        for year in years:
            ytag = f"Y{year[-2:]}"
            for cat in cats_analysis:
                channel = f"{ytag}_{cat}"
                keep = cat in cats.split(',')
                mask_params.append(f"mask_{channel}={0 if keep else 1}")

        return ",".join(mask_params)

    def run(self):
        current_cat, current_point = self._current_branch_info()
        current_dir = "_".join(current_cat.split(",")) if current_cat != self.cat else ""
        
        if self.variable == '':
            fitFolderName = f'runFits_mu_fiducial'
        else:
            fitFolderName = f'runFits_{self.variable}'
                    
        config = self.get_input_config()
        output_dir = self.get_output_dir() 

        outdir = f"outdir_{self.year}_{self.variable}" if self.variable != '' else f"outdir_{self.year}"
            
        if self.variable == '':
            datacard_path = os.path.join(output_dir, 'Combine', outdir, f'Datacard_{self.year}.root')
        else:
            datacard_path = os.path.join(output_dir, 'Combine', outdir, f'Datacard_{self.variable}_{self.year}.root')
            
        cwd = os.getcwd()
        if self.batch_flavor == "slurm/psi":
            # Have to use /scratch/batch_username/ for slurm/psi
            if "/work" in output_dir:
                execute_command([f'mkdir -p {output_dir}/Combine/{fitFolderName}/asimov'], shell=True)
            else:   
                execute_command([f'xrdfs root://t3dcachedb03.psi.ch:1094/ mkdir -p {output_dir}/Combine/{fitFolderName}/asimov'], shell=True)

            os.environ["TARGET_PATH"] = f"/scratch/{os.environ['USER']}/{os.environ['SLURM_JOB_ID']}"
            execute_command([f'mkdir -p $TARGET_PATH/Combine/{fitFolderName}/asimov'], shell=True)
            os.chdir(os.path.join(os.environ["TARGET_PATH"], 'Combine', fitFolderName, 'asimov'))
        else:
            execute_command([f'mkdir -p {output_dir}/Combine/{outdir}/{fitFolderName}/asimov'], shell=True)
            execute_command([f'mkdir -p {output_dir}/Combine/{outdir}/{fitFolderName}/asimov/{current_dir}/{self.freeze}'], shell=True)
            os.chdir(os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'asimov', current_dir, self.freeze))

        first_output = os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'asimov', self.freeze)

        def check_pdf_idx(param):
            # Run the ROOT command
            command = f'root -l -q \'{os.environ["ANALYSIS_PATH"]}/Combine/checkPdfIdx.C("{first_output}/higgsCombinefirstStep_{param}.MultiDimFit.mH{HIGGS_MASS}.root")\''
            
            # Execute the command and capture the output
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            
            # Get the output and check for errors
            pdfIdx = result.stdout.strip()
            
            if result.returncode != 0:
                print("Error executing the command:", result.stderr)
                return None
            
            if pdfIdx.startswith("Processing"):
                pdfIdx = pdfIdx.split('X', 1)[-1]  # Split on the first 'X'
            
            # Remove the last comma
            pdfIdx = pdfIdx.rstrip(',')

            # Print the final result
            print(pdfIdx)
            return pdfIdx

        if self.variable != 'MH':
            pdfIdx = check_pdf_idx(current_cat)

        firstStepPath = os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'asimov', f"higgsCombinefirstStep_{self.cat}.MultiDimFit.mH{HIGGS_MASS}.root")

        if self.variable in ['', 'tuto']:
            arguments = [
                "combine",
                "-M", "MultiDimFit",
                "-d", firstStepPath,
                # "--snapshotName", "MultiDimFit",
                "--freezeParameters", "MH",
                "-m", f"{HIGGS_MASS}",
                "-n", f"AsimovPostFitScanFit_{current_cat}.POINTS.{current_point}.{current_point}",
                "--cminDefaultMinimizerStrategy=0",
                "--cminFallbackAlgo", "Minuit2,Migrad,1:10",
                "--algo", "grid",
                "--points", f"{int(self.nPoints)}",
                "--expectSignal", "1",
                "--X-rtd", "MINIMIZER_freezeDisassociatedParams",
                "--X-rtd", "MINIMIZER_multiMin_hideConstants",
                "--X-rtd", "MINIMIZER_multiMin_maskConstraints",
                "--X-rtd", "MINIMIZER_multiMin_maskChannels=2",
                "-t", "-1",
                "-P", "r",
                "--firstPoint", f"{current_point}",
                "--lastPoint", f"{current_point}",
                "--floatOtherPOIs", "1",
                "--alignEdges", "1",
                "--setParameterRanges", f"{config['combine_fit']['setParameterRange']}",
                "--saveSpecifiedNuis", "all"
            ]

            if convert_boolean_string(self.set_pdfidx_inclusives):
                arguments.append("--setParameters")
                arguments.append(f"""{pdfIdx}""")
            else:
                arguments.append("--setParameters")
                arguments.append("r=1")

        elif self.variable == 'MH':
            arguments = [
                "combineTool.py",
                "-M", "MultiDimFit",
                "-d", firstStepPath,
                "-m", f"{HIGGS_MASS}",
                "-n", f"AsimovPostFitScanFit_{self.cat}.POINTS.{current_point}.{current_point}",
                "--cminDefaultMinimizerStrategy=0",
                "--algo", "grid",
                "--points", f"{int(self.nPoints)}",
                "--expectSignal", "1",
                "--X-rtd", "MINIMIZER_freezeDisassociatedParams",
                "--X-rtd", "MINIMIZER_multiMin_hideConstants",
                "--X-rtd", "MINIMIZER_multiMin_maskConstraints",
                "--X-rtd", "MINIMIZER_multiMin_maskChannels=2",
                "-t", "-1",
                "-P", f"{self.cat}",
                "--firstPoint", f"{current_point}",
                "--lastPoint", f"{current_point}",
                "--saveFitResult",
                "--floatOtherPOIs", "1",
                "--alignEdges", "1",
                "--snapshotName", "MultiDimFit",
                "--setParameterRanges", f"{config['combine_fit']['setParameterRange']}",
            ]   

            freezeParameters = "r"
            # freeze all syst
            if self.freeze == "allConstrainedNuisances":
                freezeParameters = f"{freezeParameters},allConstrainedNuisances"
            # freeze nuisance group defined in the datacard.txt
            elif self.freeze != "":
                arguments.extend(["--freezeNuisanceGroups", ",".join(self.freeze.split("_"))])

            arguments.extend(["--freezeParameters", freezeParameters])

            setParameters = "r=1.0"
            # mask channels/cats
            if self.cats != "":
                mask = self.make_channel_mask_params(self.cats)
                setParameters = f"{setParameters},{mask}"
            arguments.extend(["--setParameters", setParameters])


                

            if current_dir != current_cat:
                pass

        else:
            pdf_indices = combineVariableDict(self.variable, self.year)['pdfIndeces']
            cache_key = (datacard_path,)
            if cache_key not in _PDFINDEX_CACHE and os.path.exists(datacard_path):
                datacard_pdf_indices = extract_pdf_indices(datacard_path)
                if datacard_pdf_indices:
                    _PDFINDEX_CACHE[cache_key] = datacard_pdf_indices
            if cache_key in _PDFINDEX_CACHE:
                pdf_indices = _PDFINDEX_CACHE[cache_key]
            paramStr = ",".join(combineVariableDict(self.variable, self.year)['paramStr'])
            set_param_string = paramStr
            if pdfIdx:
                set_param_string = f"{set_param_string},{pdfIdx}"

            arguments = [
                "combineTool.py",
                "-M", "MultiDimFit",
                "-d", firstStepPath,
                "-m", f"{HIGGS_MASS}",
                "-n", f"AsimovPostFitScanFit_{current_cat}.POINTS.{current_point}.{current_point}",
                "--cminDefaultMinimizerStrategy=0",
                "--cminFallbackAlgo", "Minuit2,Migrad,1:10",
                "--algo", "grid",
                "--points", f"{int(self.nPoints)}",
                "--expectSignal", "1",
                "--X-rtd", "MINIMIZER_freezeDisassociatedParams",
                "--X-rtd", "MINIMIZER_multiMin_hideConstants",
                "--X-rtd", "MINIMIZER_multiMin_maskConstraints",
                "--X-rtd", "MINIMIZER_multiMin_maskChannels=2",
                "-t", "-1",
                "-P", f"{current_cat}",
                "--firstPoint", f"{current_point}",
                "--lastPoint", f"{current_point}",
                "--saveFitResult",
                "--floatOtherPOIs", "1",
                "--alignEdges", "1",
                "--snapshotName", "MultiDimFit",
                "--setParameters", set_param_string,
            ]
            if self.freeze == "allConstrainedNuisances":
                arguments.extend(["--freezeParameters", "MH,allConstrainedNuisances"])
            elif self.freeze != "":
                arguments.extend(["--freezeParameters", "MH"])
                arguments.extend(["--freezeNuisanceGroups", self.freeze])
            else:
                arguments.extend(["--freezeParameters", "MH"])

            arguments.extend(_scan_parameter_range_args(config, current_cat))
            arguments.extend(_save_specified_index_args(pdf_indices))
        command = arguments
        print(' '.join(command))
        try:
            result = subprocess.run(command, check=True, text=True, capture_output=True)
            print("Script output:", result.stdout)
            print("Script executed successfully.")
        except subprocess.CalledProcessError as e:
            print("Error executing script:", e.stderr)
            raise
            
        # Copy the files back to pnfs if we are on slurm/psi
        if self.batch_flavor == "slurm/psi":
            # Have to copy over the output to the final directory
            # Don't forget to VOMS!
            if "/work" in output_dir:
                slurm_copy_command = [
                    'cp', '-rf',
                    f"{os.environ['TARGET_PATH']}/Combine/",
                    output_dir
                ]
            else:
                slurm_copy_command = [
                    'xrdcp', '-rf',
                    f"{os.environ['TARGET_PATH']}/Combine/",
                    'root://t3dcachedb03.psi.ch:1094//'+output_dir
                ]
            print(slurm_copy_command)
            execute_command(slurm_copy_command)
            # Clean up the temporary directory
            shutil.rmtree(os.environ["TARGET_PATH"])
            
        os.chdir(cwd)
        
        
class CreateAsimovFit(MultiYearTask, law.LocalWorkflow, HTCondorWorkflow, SlurmWorkflow):

    group = law.Parameter(default="Syst,Stat")
    set_pdfidx_inclusives = law.Parameter(default=False)
    cats = law.Parameter(default="")

    def workflow_requires(self):
        workflow_reqs = super().workflow_requires()

        tasks = {}

        if workflow_reqs:
            tasks.update(workflow_reqs)

        output_dir = self.get_output_dir()

        config = self.get_input_config()
        output_dir = self.get_output_dir()
        config_combine = config["combine_fit"]

        kwargs = {
            "nPoints": config_combine["asimov_numPoints"],
            "workflow": config_combine["execution"],
            "slurm_partition": config_combine["batchPartition"],
            "slurm_memory": config_combine["batchMemory"],
            "slurm_max_runtime": config_combine["batchMaxRuntime"],
            "htcondor_partition": config_combine["batchPartition"],
            "htcondor_memory": config_combine["batchMemory"],
            "htcondor_max_runtime": config_combine["batchMaxRuntime"],
        }

        cat = "MH" if self.variable == "MH" else "r"

        for i, syst in enumerate(self.group.split(',')):
            if i == len(self.group.split(','))-1:
                freeze = "allConstrainedNuisances"
            else:
                freeze = '_'.join(self.group.split(",")[:i])

            tasks[f"AsimovFitCategory{syst}"] = AsimovFitCategorySyst.req(
                self,
                output_dir=output_dir,
                cat=cat,
                freeze=freeze,
                set_pdfidx_inclusives=self.set_pdfidx_inclusives,
                **kwargs,
            )

        return tasks

    def _requires_single(self):
        return self.workflow_requires()

    
    def create_branch_map(self):
        # map branch indexes to ascii numbers from 97 to 122 ("a" to "z")        
        branch_list = [0]
        
        branch_map = {i: branch for i, branch in enumerate(branch_list)}
        return branch_map

    # def output(self):
    #     config = self.get_input_config()
    #     output_dir = self.get_output_dir()
    #     current_dir = "_".join(self.cats.split(","))

    #     if self.variable == '':
    #         fitFolderName = f'runFits_mu_fiducial'
    #     else:
    #         fitFolderName = f'runFits_{self.variable}'
            
    #     output = []

    #     outdir = f'outdir_{self.year}_{self.variable}' if self.variable != '' else '' 
            
    #     output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'asimov', 'scans', current_dir)]

    #     group = self.group
    #     for i,syst in enumerate(group.split(',')):
    #         if i == len(self.group.split(','))-1:
    #             freeze = "allConstrainedNuisances"
    #         else:
    #             freeze = '_'.join(self.group.split(",")[:i])
    #         if self.variable in ['','tuto', 'MH']:
    #             cat = ["MH"] if self.variable == "MH" else ["r"]
    #         else:
    #             cat = combineVariableDict(self.variable, self.year)['paramStrNoOne']

    #         for c in cat:
    #             if i == 0:
    #                 output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'asimov', 'scans', current_dir, f'scan_{c}_{"_".join(group.split(","))}.root')]
    #                 output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'asimov', 'scans', current_dir, f'scan_{c}_{"_".join(group.split(","))}.pdf')]
    #                 output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'asimov', 'scans', current_dir, f'scan_{c}_{"_".join(group.split(","))}.png')]
    #             output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'asimov', current_dir, freeze, f'higgsCombineAsimovPostFitScanFit_{c}.root')]
        

    #     outputFileTargets = []
                
    #     for _, current_output_path in enumerate(output):
    #         outputFileTargets.append(law.LocalFileTarget(current_output_path))
            
    #     # print(outputFileTargets)

    #     return outputFileTargets

    def output(self):
        output_dir = self.get_output_dir()
        current_dir = "_".join(self.cats.split(","))

        if self.variable == "":
            fitFolderName = "runFits_mu_fiducial"
        else:
            fitFolderName = f"runFits_{self.variable}"

        outdir = (
            f"outdir_{self.year}_{self.variable}"
            if self.variable != ""
            else ""
        )

        base_dir = os.path.join(
            output_dir,
            "Combine",
            outdir,
            fitFolderName,
            "asimov",
        )

        scan_dir = os.path.join(
            base_dir,
            "scans",
            current_dir,
        )

        if self.variable in ["", "tuto", "MH"]:
            pois = ["MH"] if self.variable == "MH" else ["r"]
        else:
            pois = combineVariableDict(
                self.variable,
                self.year,
            )["paramStrNoOne"]

        outputs = {
            "scan_dir": law.LocalDirectoryTarget(scan_dir),
            "scans": {},
            "plots": {},
        }

        groups = self.group.split(",")

        for i, syst in enumerate(groups):

            if i == len(groups) - 1:
                freeze = "allConstrainedNuisances"
            else:
                freeze = "_".join(groups[:i])

            # Give the important scans semantic names
            if freeze == "":
                scan_type = "total"
            elif freeze == "allConstrainedNuisances":
                scan_type = "stat"
            else:
                scan_type = freeze

            outputs["scans"][scan_type] = {}

            for poi in pois:
                path = os.path.join(
                    base_dir,
                    current_dir,
                    freeze,
                    f"higgsCombineAsimovPostFitScanFit_{poi}.root",
                )

                outputs["scans"][scan_type][poi] = \
                    law.LocalFileTarget(path)

        # plot outputs are produced only once
        group_name = "_".join(groups)

        for poi in pois:
            outputs["plots"][poi] = {}

            for ext in ("root", "pdf", "png"):
                path = os.path.join(
                    scan_dir,
                    f"scan_{poi}_{group_name}.{ext}",
                )

                outputs["plots"][poi][ext] = \
                    law.LocalFileTarget(path)

        return outputs

    def run(self):
        
        if self.variable == '':
            fitFolderName = f'runFits_mu_fiducial'
        else:
            fitFolderName = f'runFits_{self.variable}'

        outdir = f'outdir_{self.year}_{self.variable}' if self.variable != '' else ''
                    
        config = self.get_input_config()
        output_dir = self.get_output_dir() 
        current_dir = "_".join(self.cats.split(","))
            
        cwd = os.getcwd()
        
        if self.batch_flavor == "slurm/psi":
            # Have to use /scratch/batch_username/ for slurm/psi
            if "/work" in output_dir:
                execute_command([f'mkdir -p {output_dir}/Combine/{fitFolderName}/asimov/scans'], shell=True)
            else:   
                execute_command([f'xrdfs root://t3dcachedb03.psi.ch:1094/ mkdir -p {output_dir}/Combine/{fitFolderName}/asimov/scans'], shell=True)

            os.environ["TARGET_PATH"] = f"/scratch/{os.environ['USER']}/{os.environ['SLURM_JOB_ID']}"
            execute_command([f'mkdir -p $TARGET_PATH/Combine/{fitFolderName}/asimov/scans'], shell=True)
            os.chdir(os.path.join(os.environ["TARGET_PATH"], 'Combine', fitFolderName, 'asimov'))
        else:
            execute_command([f'mkdir -p {output_dir}/Combine/{outdir}/{fitFolderName}/asimov/scans/{current_dir}'], shell=True)
            os.chdir(os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'asimov', current_dir))
        
        if self.variable in ['', 'tuto', 'MH']:
            cats = ["MH"] if self.variable == "MH" else ["r"]
        else:
            cats = combineVariableDict(self.variable, self.year)['paramStrNoOne']
        
        if self.batch_flavor == "slurm/psi":
            # Have to copy over the input to the JOB directory
            # Don't forget to VOMS!
            if "/work" in output_dir:
                slurm_copy_command = [
                    'cp', '-rf',
                    f'{output_dir}/Combine/{fitFolderName}/asimov',
                    f"{os.environ['TARGET_PATH']}/Combine/{fitFolderName}"
                ]
            else:
                slurm_copy_command = [
                    'xrdcp', '-rf',
                    'root://t3dcachedb03.psi.ch:1094//'+f'{output_dir}/Combine/{fitFolderName}/asimov',
                    f"{os.environ['TARGET_PATH']}/Combine/{fitFolderName}"
                ]
            print(slurm_copy_command)
            execute_command(slurm_copy_command)
        
        for cat in cats:
            freeze_list = []
            # hadd the files for each category and freeze combination
            for i,syst in enumerate(self.group.split(',')):
                if i == len(self.group.split(','))-1:
                    freeze = "allConstrainedNuisances"
                else:
                    freeze = '_'.join(self.group.split(",")[:i])

                freeze_list.append(freeze)
        
                arguments = [
                    "hadd", "-f",
                    f"{os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'asimov', current_dir, freeze, f'higgsCombineAsimovPostFitScanFit_{cat}.root')}"
                ]
                for i in range(config["combine_fit"]["asimov_numPoints"]):
                    arguments.append(os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'asimov', current_dir, freeze, f'higgsCombineAsimovPostFitScanFit_{cat}.POINTS.{i}.{i}.MultiDimFit.mH{HIGGS_MASS}.root'))

                if self.batch_flavor == "slurm/psi":
                    arguments = [
                        "hadd", "-f",
                        f"{os.path.join(os.environ['TARGET_PATH'], 'Combine', fitFolderName, 'asimov', current_dir, freeze, f'higgsCombineAsimovPostFitScanFit_{cat}.root')}"
                    ]
                    for i in range(config["combine_fit"]["asimov_numPoints"]):
                        arguments.append(os.path.join(os.environ['TARGET_PATH'], 'Combine', fitFolderName, 'asimov', current_dir, freeze, f'higgsCombineAsimovPostFitScanFit_{cat}.POINTS.{i}.{i}.MultiDimFit.mH{HIGGS_MASS}.root'))
                command = arguments
                # print(command)
                try:
                    result = subprocess.run(command, check=True, text=True, capture_output=True)
                    print("Script output:", result.stdout)
                    print("Script executed successfully.")
                except subprocess.CalledProcessError as e:
                    print("Error executing script:", e.stderr)
                    raise
                    
                

            # change to the scans directory
            if self.batch_flavor == "slurm/psi":
                os.chdir(os.path.join(os.environ["TARGET_PATH"], 'Combine', fitFolderName, 'asimov', 'scans', current_dir))
            else:
                os.chdir(os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'asimov', 'scans', current_dir))

            arguments = [
                "python3", 
                # f"{os.environ['CMSSW_BASE']}/bin/{os.environ['SCRAM_ARCH']}/plot1DScan.py",
                os.path.join(os.environ["ANALYSIS_PATH"],"Plots/plot1DScan.py"),
                # "plot1DScan.py",
                os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'asimov', current_dir, freeze_list[0], f'higgsCombineAsimovPostFitScanFit_{cat}.root'),
                "-o", f'scan_{cat}_{"_".join(self.group.split(","))}',
                "--POI", f"{cat}",
                "--main-label", "Expected",
                "--translate", os.path.join(os.environ["ANALYSIS_PATH"], 'Combine', 'pois.json'),
                "--breakdown", self.group,
                "--x-min", str(config["combine_fit"].get("xMin", 124)),
                "--x-max", str(config["combine_fit"].get("xMax", 126)),
                "--others", 
            ]

            arguments += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'asimov', current_dir, freeze, f'higgsCombineAsimovPostFitScanFit_{cat}.root')+f":{'Stat only' if freeze=='allConstrainedNuisances' else 'freeze '+'+'.join(freeze.split('_'))}:{i+2}" for i, (freeze, name) in enumerate(zip(freeze_list[1:], self.group.split(',')[1:]))]

            command = arguments
            print(' '.join(command))
            try:
                result = subprocess.run(command, check=True, text=True, capture_output=True)
                print("Script output:", result.stdout)
                print("Script executed successfully.")
            except subprocess.CalledProcessError as e:
                print("Error executing script:", e.stderr)
    
        # Copy the files back to pnfs if we are on slurm/psi
        if self.batch_flavor == "slurm/psi":
            # Have to copy over the output to the final directory
            # Don't forget to VOMS!
            if "/work" in output_dir:
                slurm_copy_command = [
                    'cp', '-rf',
                    f"{os.environ['TARGET_PATH']}/Combine/",
                    output_dir
                ]
            else:
                slurm_copy_command = [
                    'xrdcp', '-rf',
                    f"{os.environ['TARGET_PATH']}/Combine/",
                    'root://t3dcachedb03.psi.ch:1094//'+output_dir
                ]
            print(slurm_copy_command)
            execute_command(slurm_copy_command)
            # Clean up the temporary directory
            shutil.rmtree(os.environ["TARGET_PATH"])
            
        os.chdir(cwd)


class CreateAsimovFitWrapper(MultiYearTask):

    set_pdfidx_inclusives = law.Parameter(default=False)
    do_per_cat = luigi.BoolParameter(default=False, description="Run per-category fits instead of inclusive fits")

    # def _requires_single(self):
    #     config = self.get_input_config()
    #     output_dir = self.get_output_dir()

    #     groups = config["combine_fit"].get("group", ["Syst,Stat"])
    #     if isinstance(groups, str):
    #         groups = [groups]

    #     tasks = {}

    #     for cats in ([""] + self.get_cats().split(",")):
    #         # do Syst,Stat and syst groups splitting Zmmg,Zee,Smearing,Other,Stat
    #         for group in groups:
    #             label = group.replace(",", "_")
    #             tasks[f"CreateAsimovFit_{label}_{cats}"] = CreateAsimovFit.req(
    #                 self,
    #                 years=self.years,
    #                 output_dir=output_dir,
    #                 group=group,
    #                 set_pdfidx_inclusives=self.set_pdfidx_inclusives,
    #                 workflow=self.batch_flavor,
    #                 cats=cats
    #             )

    #             if cats != "":
    #                 break



    #     return tasks
    
    def _requires_single(self):
        config = self.get_input_config()
        output_dir = self.get_output_dir()

        groups = config["combine_fit"].get(
            "group",
            ["Syst,Stat"],
        )

        if isinstance(groups, str):
            groups = [groups]

        tasks = {
            "inclusive": {},
            "categories": {},
        }

        # Inclusive fits: preserve all configured groups
        for group in groups:
            label = group.replace(",", "_")

            tasks["inclusive"][label] = CreateAsimovFit.req(
                self,
                years=self.years,
                output_dir=output_dir,
                group=group,
                set_pdfidx_inclusives=self.set_pdfidx_inclusives,
                workflow=self.batch_flavor,
                cats="",
            )

        if self.do_per_cat:
            # Per-category fits only need the first group,
            # matching your previous behavior.
            group = groups[0]

            for cat in self.get_cats().split(","):
                cat = cat.strip()

                if not cat:
                    continue

                tasks["categories"][cat] = CreateAsimovFit.req(
                    self,
                    years=self.years,
                    output_dir=output_dir,
                    group=group,
                    set_pdfidx_inclusives=self.set_pdfidx_inclusives,
                    workflow=self.batch_flavor,
                    cats=cat,
                )

        return tasks

    def output(self):
        return self.input()

    def run(self):
        return True


class CreateAsimovFitPerCat(MultiYearTask):
    include_stat_error = luigi.BoolParameter(default=True, description="Include stat-only scan inputs and print stat/syst breakdown")
    show_values = luigi.BoolParameter(default=True, description="Draw numerical fit values on each summary row")
    is_per_year = luigi.BoolParameter(default=False, description="Show per year breakdown in the summary plot")
    set_pdfidx_inclusives = law.Parameter(default=False)

    def _per_year_rows(self):
        if self.year not in yearMap:
            raise ValueError(
                f"No per-year expansion is configured for '{self.year}'. "
                f"Available entries: {sorted(yearMap)}"
            )
        # Keep the combined result last and avoid duplicating single-year entries.
        return [year for year in yearMap[self.year] if year != self.year] + [self.year]

    def _requires_single(self):
        if self.is_per_year:
            tasks = {}
            for year in self._per_year_rows():
                tasks[year] = CreateAsimovFitWrapper.req(
                    self,
                    years=year,
                    output_dir=self.get_output_dir(),
                    set_pdfidx_inclusives=self.set_pdfidx_inclusives,
                    do_per_cat=False
                )
            return tasks
        return {self.year: CreateAsimovFitWrapper.req(
            self,
            years=self.years,
            output_dir=self.get_output_dir(),
            set_pdfidx_inclusives=self.set_pdfidx_inclusives,
            do_per_cat=True
        )}

    def _scan_categories(self):
        return [cat.strip() for cat in self.get_cats().split(",") if cat.strip()]

    def _fit_folder_name(self):
        if self.variable == '':
            return "runFits_mu_fiducial"
        return f"runFits_{self.variable}"

    def _outdir(self):
        return f"outdir_{self.year}_{self.variable}" if self.variable != '' else f"outdir_{self.year}"

    def _summary_dir(self):
        return os.path.join(
            self.get_output_dir(),
            "Combine",
            self._outdir(),
            self._fit_folder_name(),
            "asimov",
            "scans",
            "perCat",
        )

    def _output_base(self):
        suffix = "stat_syst" if self.include_stat_error else "total"
        breakdown = "perYear" if self.is_per_year else "perCat"
        return os.path.join(self._summary_dir(), f"scan_MH_{breakdown}_{suffix}")

    def output(self):
        output = []
        output += [self._output_base() + ext for ext in [".root", ".pdf", ".png"]]
        return [law.LocalFileTarget(path) for path in output]



    def _label_for_cat(self, cat):
        label = re.sub(r"([A-Za-z]+)cat([0-9]+)$", r"\1 cat\2", cat)
        label = label.replace("notEBEB", "notEBEB ")
        return f"CMS H#gamma#gamma {label}"

    def _unwrap_workflow_input(self, value):
        if isinstance(value, dict) and "collection" in value:
            targets = value["collection"].targets
            if isinstance(targets, dict):
                return next(iter(targets.values()))

            if isinstance(targets, (list, tuple)):
                return targets[0]
            return targets
        return value

    def _inclusive_scan_argument(self, wrapper_input, group_label, label):
        """Build a plotter scan argument from one wrapper's inclusive fit."""
        try:
            inclusive_input = self._unwrap_workflow_input(
                wrapper_input["inclusive"][group_label]
            )
            total_scan = inclusive_input["scans"]["total"]["MH"].path
        except KeyError as exc:
            raise RuntimeError(
                f"Could not find inclusive total MH scan for '{label}' and "
                f"group '{group_label}'. Available input structure: {wrapper_input}"
            ) from exc

        scan_argument = f"{label}:{total_scan}"
        if self.include_stat_error:
            try:
                stat_scan = inclusive_input["scans"]["stat"]["MH"].path
            except KeyError as exc:
                raise RuntimeError(
                    f"Could not find inclusive stat-only MH scan for '{label}'. "
                    f"Available input structure: {inclusive_input}"
                ) from exc
            scan_argument += f":{stat_scan}"

        return scan_argument

    def run(self):
        if self.variable != "MH":
            raise RuntimeError(
                "CreateAsimovFitPerCat is currently intended for variable=MH scans"
            )

        cwd = os.getcwd()

        try:
            # Create output directory and move there
            execute_command(
                [f"mkdir -p {self._summary_dir()}"],
                shell=True,
            )
            os.chdir(self._summary_dir())

            # Inputs from CreateAsimovFitWrapper
            inputs = self.input()

            # In category mode, _requires_single() namespaces the wrapper under
            # the current year.  The plotting code below consumes the wrapper
            # output itself (the dict containing "inclusive" and "categories").
            if not self.is_per_year:
                if self.year not in inputs:
                    raise RuntimeError(
                        f"Year '{self.year}' not found in self.input(). Available "
                        f"keys: {list(inputs.keys())}"
                    )
                inputs = inputs[self.year]

            cats = self._scan_categories()
            config = self.get_input_config()

            # Get configured fit groups
            groups = config["combine_fit"].get(
                "group",
                ["Syst,Stat"],
            )

            if isinstance(groups, str):
                groups = [groups]

            # The per-category tasks use the first configured group
            main_group_label = groups[0].replace(",", "_")

            if self.is_per_year:
                plot_rows = self._per_year_rows()
                split_after = len(plot_rows) - 1
                band_from = len(plot_rows)
            else:
                plot_rows = cats + [self.year]
                split_after = len(cats)
                band_from = len(cats) + 1

            arguments = [
                "python3",
                os.path.join(
                    os.environ["ANALYSIS_PATH"],
                    "Plots",
                    "plotAsimovScanSummary.py",
                ),
                "--POI",
                "MH",
                "--output",
                self._output_base(),
                "--band-from",
                str(band_from),
                "--split-after",
                str(split_after),
                "--x-title",
                "m_{H} (GeV)",
                "--x-min",
                str(
                    config["combine_fit"].get(
                        "xMin",
                        124,
                    )
                ),
                "--x-max",
                str(
                    config["combine_fit"].get(
                        "xMax",
                        126,
                    )
                ),
                "--cms-label",
                "Internal",
                "--lumi-label",
                get_lumi_label(self.year),
            ]

            if self.show_values:
                arguments.append("--show-values")

            if self.include_stat_error:
                arguments.append("--show-breakdown")

            if self.is_per_year:
                # Each entry is the output of one inclusive-only wrapper.  Put the
                # combined fit last so it also defines the uncertainty band.
                for year in plot_rows:
                    if year not in inputs:
                        raise RuntimeError(
                            f"Year '{year}' not found in self.input(). Available "
                            f"years: {list(inputs.keys())}"
                        )
                    label = (
                        f"CMS H#gamma#gamma {year}"
                        if year != self.year
                        else f"CMS H#gamma#gamma {self.year} combined"
                    )
                    arguments += [
                        "--scan",
                        self._inclusive_scan_argument(
                            inputs[year], main_group_label, label
                        ),
                    ]
            else:
                # --------------------------------------------------------
                # Per-category scans
                # --------------------------------------------------------
                for cat in cats:
                    if cat not in inputs["categories"]:
                        raise RuntimeError(
                            f"Category '{cat}' not found in self.input()['categories']. "
                            f"Available categories: {list(inputs['categories'].keys())}"
                        )

                    cat_input = self._unwrap_workflow_input(inputs["categories"][cat])

                    try:
                        total_scan = cat_input["scans"]["total"]["MH"].path
                    except KeyError as exc:
                        raise RuntimeError(
                            f"Could not find total MH scan for category '{cat}'. "
                            f"Available input structure: {cat_input}"
                        ) from exc

                    scan_argument = f"{self._label_for_cat(cat)}:{total_scan}"

                    if self.include_stat_error:
                        try:
                            stat_scan = cat_input["scans"]["stat"]["MH"].path
                        except KeyError as exc:
                            raise RuntimeError(
                                f"Could not find stat-only MH scan for category '{cat}'. "
                                f"Available input structure: {cat_input}"
                            ) from exc
                        scan_argument += f":{stat_scan}"

                    arguments += ["--scan", scan_argument]

            # ------------------------------------------------------------
            # Inclusive scan
            # ------------------------------------------------------------
            if not self.is_per_year:
                arguments += [
                    "--scan",
                    self._inclusive_scan_argument(
                        inputs,
                        main_group_label,
                        f"CMS H#gamma#gamma {self.year}",
                    ),
                ]

            # ------------------------------------------------------------
            # Run plotting script
            # ------------------------------------------------------------
            print("Running command:")
            print(" ".join(arguments))

            result = subprocess.run(
                arguments,
                check=True,
                text=True,
                capture_output=True,
            )

            print("Script output:")
            print(result.stdout)

            if result.stderr:
                print("Script stderr:")
                print(result.stderr)

            print("Script executed successfully.")

        except subprocess.CalledProcessError as e:
            print("Error executing plotAsimovScanSummary.py")

            if e.stdout:
                print("stdout:")
                print(e.stdout)

            if e.stderr:
                print("stderr:")
                print(e.stderr)

            raise

        finally:
            os.chdir(cwd)
    
class AsimovImpactFirstStep(Task, law.LocalWorkflow, HTCondorWorkflow, SlurmWorkflow): #(law.Task): #(Task, HTCondorWorkflow, law.LocalWorkflow):

    # def requires(self):
    def workflow_requires(self):
        workflow_reqs = super().workflow_requires()

        tasks = {}

        if workflow_reqs:
            tasks.update(workflow_reqs)
        
        config = self.get_input_config()
        output_dir = self.get_output_dir()
        
        impactConfig = config["combine_impacts"]    
        
        tasks["RunT2WS"] = RunText2Workspace.req(self, output_dir=output_dir)
        tasks["CreateAsimovFitFirstStep"] = CreateAsimovFitFirstStep.req(self, output_dir=output_dir, )
        
        return tasks

    def create_branch_map(self):
        branch_map = {i: current_branch for i, current_branch in enumerate([0])}
        return branch_map

    def output(self):        
        config = self.get_input_config()
        output_dir = self.get_output_dir()
        outdir = f'outdir_{self.year}_{self.variable}' if self.variable != '' else ''

        if self.variable == '':
            fitFolderName = f'runFits_mu_fiducial'
        else:
            fitFolderName = f'runFits_{self.variable}'

            
        # output = [os.path.join(output_dir, 'Combine', outdir, fitFolderName)]
        output = [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'impact')]

        output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'impact', f'higgsCombine_initialFit_Test.MultiDimFit.mH{HIGGS_MASS}.root')]
        
        outputFileTargets = []
                
        for _, current_output_path in enumerate(output):
            outputFileTargets.append(law.LocalFileTarget(current_output_path))
            
        # print(outputFileTargets)

        return outputFileTargets

    def run(self):
       
        if self.variable == '':
            fitFolderName = f'runFits_mu_fiducial'
            
        else:
            fitFolderName = f'runFits_{self.variable}'
                  
        config = self.get_input_config()
        output_dir = self.get_output_dir()
        outdir = f'outdir_{self.year}_{self.variable}' if self.variable != '' else f'outdir_{self.year}' 
            
        if self.variable == '':
            datacard_path = os.path.join(output_dir, 'Combine', outdir, f'Datacard_{self.year}.root')
        else:
            datacard_path = os.path.join(output_dir, 'Combine', outdir, f'Datacard_{self.variable}_{self.year}.root')
        
        cwd = os.getcwd()
        if self.batch_flavor == "slurm/psi":
            # Have to use /scratch/batch_username/ for slurm/psi
            if "/work" in output_dir:
                execute_command([f'mkdir -p {output_dir}/Combine/{outdir}/{fitFolderName}/impact'], shell=True)
            else:   
                execute_command([f'xrdfs root://t3dcachedb03.psi.ch:1094/ mkdir -p {output_dir}/Combine/{outdir}/{fitFolderName}/impact'], shell=True)

            os.environ["TARGET_PATH"] = f"/scratch/{os.environ['USER']}/{os.environ['SLURM_JOB_ID']}"
            execute_command([f'mkdir -p $TARGET_PATH/Combine/{outdir}/{fitFolderName}/impact'], shell=True)
            os.chdir(os.path.join(os.environ["TARGET_PATH"], 'Combine', outdir, fitFolderName, 'impact'))
        else:
            execute_command([f'mkdir -p {output_dir}/Combine/{outdir}/{fitFolderName}/impact'], shell=True)
            os.chdir(os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'impact'))

        first_output = os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'asimov')
        if self.variable in ['', 'tuto']:
            pdf_idx_param = "r"
        elif self.variable == "MH":
            pdf_idx_param = "MH"
        else:
            pdf_idx_param = combineVariableDict(self.variable, self.year)['paramStrNoOne'][0]

        def check_pdf_idx(param):
            command = f'root -l -q \'{os.environ["ANALYSIS_PATH"]}/Combine/checkPdfIdx.C("{first_output}/higgsCombinefirstStep_{param}.MultiDimFit.mH{HIGGS_MASS}.root")\''
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            pdfIdx = result.stdout.strip()
            if result.returncode != 0:
                print("Error executing the command:", result.stderr)
                return None
            if pdfIdx.startswith("Processing"):
                pdfIdx = pdfIdx.split('X', 1)[-1]
            pdfIdx = pdfIdx.rstrip(',')
            print(pdfIdx)
            return pdfIdx

        pdfIdx = None
        first_step_path = os.path.join(first_output, f"higgsCombinefirstStep_{pdf_idx_param}.MultiDimFit.mH{HIGGS_MASS}.root")
        if os.path.exists(first_step_path):
            pdfIdx = check_pdf_idx(pdf_idx_param)
        else:
            print(f"First-step file not found for PDF index extraction: {first_step_path}")
        
        if self.variable in ['', "tuto"]:
            set_param_string = "r=1"
            if pdfIdx:
                set_param_string = f"{set_param_string},{pdfIdx}"
            arguments = [
                "combineTool.py",
                "-M", "Impacts",
                "-d", datacard_path,
                "--doInitialFit",
                "--robustFit", "1",
                "--freezeParameters", "MH",
                "-m", "125.38",
                "--cminDefaultMinimizerStrategy=0",
                "--X-rtd", "MINIMIZER_freezeDisassociatedParams",
                "--X-rtd", "MINIMIZER_multiMin_hideConstants",
                "--X-rtd", "MINIMIZER_multiMin_maskConstraints",
                "--X-rtd", "MINIMIZER_multiMin_maskChannels=2",
                "-t", "-1",
                "--setParameters", f"{set_param_string}"
            ]
            command = arguments
            # print(command)
            try:
                result = subprocess.run(command, check=True, text=True, capture_output=True)
                print("Script output:", result.stdout)
                print("Script executed successfully.")
            except subprocess.CalledProcessError as e:
                print("Error executing script:", e.stderr)

        elif self.variable in ["MH"]:
                    set_param_string = f"r=1,MH={HIGGS_MASS}"
                    if pdfIdx:
                        set_param_string = f"{set_param_string},{pdfIdx}"
                    arguments = [
                        "combineTool.py",
                        "-M", "Impacts",
                        "-d", datacard_path,
                        "--doInitialFit",
                        "--robustFit", "1",
                        "--freezeParameters", "r",
                        "--redefineSignalPOIs", "MH",
                        "-m", f"{HIGGS_MASS}",
                        "--cminDefaultMinimizerStrategy=0",
                        "--cminFallbackAlgo", "Minuit2,Migrad,1:10",
                "--X-rtd", "MINIMIZER_freezeDisassociatedParams",
                        "--X-rtd", "MINIMIZER_multiMin_hideConstants",
                        "--X-rtd", "MINIMIZER_multiMin_maskConstraints",
                        "--X-rtd", "MINIMIZER_multiMin_maskChannels=2",
                        "-t", "-1",
                        "--setParameters", f"{set_param_string}",
                        "--setParameterRanges", "MH=123,127"
                    ]
                    command = arguments
                    print('AsimovImpactFirstStep: ' + ' '.join(command))
                    try:
                        result = subprocess.run(command, check=True, text=True, capture_output=True)
                        print("Script output:", result.stdout)
                        print("Script executed successfully.")
                    except subprocess.CalledProcessError as e:
                        print("Error executing script:", e.stderr)
        else:
            set_param_string = ",".join(combineVariableDict(self.variable, self.year)['paramStr'])
            if pdfIdx:
                set_param_string = f"{set_param_string},{pdfIdx}"
            arguments = [
                "combine",
                "-M", "MultiDimFit",
                "-d", datacard_path,
                "--algo", "singles",
                "--redefineSignalPOIs", f"""{",".join(combineVariableDict(self.variable, self.year)['paramStrNoOne'])}""",
                "--freezeParameters", "MH",
                "-m", f"{HIGGS_MASS}",
                "-n", "_initialFit_Test",
                "--cminDefaultMinimizerStrategy=0",
                "--cminFallbackAlgo", "Minuit2,Migrad,1:10",
                "--X-rtd", "MINIMIZER_freezeDisassociatedParams",
                "--X-rtd", "MINIMIZER_multiMin_hideConstants",
                "--X-rtd", "MINIMIZER_multiMin_maskConstraints",
                "--X-rtd", "MINIMIZER_multiMin_maskChannels=2",
                "-t", "-1",
                "--setParameters", f"{set_param_string}"
            ]
            command = arguments
            # print(command)
            try:
                result = subprocess.run(command, check=True, text=True, capture_output=True)
                print("Script output:", result.stdout)
                print("Script executed successfully.")
            except subprocess.CalledProcessError as e:
                print("Error executing script:", e.stderr)

        # Copy the files back to pnfs if we are on slurm/psi
        if self.batch_flavor == "slurm/psi":
            # Have to copy over the output to the final directory
            # Don't forget to VOMS!
            if "/work" in output_dir:
                slurm_copy_command = [
                    'cp', '-rf',
                    f"{os.environ['TARGET_PATH']}/Combine/",
                    output_dir
                ]
            else:
                slurm_copy_command = [
                    'xrdcp', '-rf',
                    f"{os.environ['TARGET_PATH']}/Combine/",
                    'root://t3dcachedb03.psi.ch:1094//'+output_dir
                ]
            print(slurm_copy_command)
            execute_command(slurm_copy_command)
            # Clean up the temporary directory
            shutil.rmtree(os.environ["TARGET_PATH"])
        
            
        os.chdir(cwd)
        
class AsimovImpactSecondStep(Task, law.LocalWorkflow, HTCondorWorkflow, SlurmWorkflow): #(law.Task): #(Task, HTCondorWorkflow, law.LocalWorkflow):
    output_dir = law.Parameter(default = '', description="Path to the output directory")
    variable = law.Parameter(default="", description="Variable to be used")
    year = law.Parameter(default='2022', description="Year")

    batch_flavor = law.Parameter(default="htcondor", description="Batch system to use")

    # def requires(self):
    def workflow_requires(self):
        workflow_reqs = super().workflow_requires()

        tasks = {}

        if workflow_reqs:
            tasks.update(workflow_reqs)
        
        config = self.get_input_config()
        output_dir = self.get_output_dir()
        
        impactConfig = config["combine_impacts"]
            
        tasks["AsimovImpactFirstStep"] = AsimovImpactFirstStep.req(
            self,
            output_dir=output_dir,  
            workflow=impactConfig["execution"], 
            slurm_partition=impactConfig['batchPartition'], 
            slurm_memory=impactConfig['batchMemory'], 
            slurm_max_runtime=impactConfig['batchMaxRuntime'], 
            htcondor_partition=impactConfig['batchPartition'], 
            htcondor_memory=impactConfig['batchMemory'], 
            htcondor_max_runtime=impactConfig['batchMaxRuntime'])
        
        return tasks

    def create_branch_map(self):

                  
        config = self.get_input_config()
        output_dir = self.get_output_dir() 

        outdir = f'outdir_{self.year}_{self.variable}' if self.variable != '' else ''
            
        if self.variable == '':
            datacard_path = os.path.join(output_dir, 'Combine', outdir, f'Datacard_{self.year}.root')
        else:
            datacard_path = os.path.join(output_dir, 'Combine', outdir, f'Datacard_{self.variable}_{self.year}.root')
    
        # IMPORTANT:
        # For differential impacts we only want *constrained nuisance parameters* (i.e. those listed in
        # ModelConfig's nuisance set). Using "all free pdf parameters" (equivalent to combineTool.py's
        # --allPars) also includes background-model / envelope parameters (e.g. env_pdf_*) that often
        # have no meaningful +/-1sigma crossing in multi-POI fits, leading to
        #   "[ERROR] Closed range without finding crossing!"
        # and missing per-parameter output root files.
        def list_nuisance_parameters(file, wsp, mc, pois, exclude_expr=""):
            wsFile = ROOT.TFile.Open(file)
            w = wsFile.Get(wsp)
            config = w.genobj(mc)

            nuis = config.GetNuisanceParameters()
            if not nuis:
                return []

            names = []
            it = nuis.createIterator()
            var = it.Next()
            while var:
                name = var.GetName()
                if name not in pois and (not var.isConstant()) and var.InheritsFrom("RooRealVar"):
                    names.append(name)
                var = it.Next()

            all_vars = w.allVars()

            # names = []
            # it = all_vars.createIterator()
            # var = it.Next()
            # while var:
            #     name = var.GetName()
            #     if (
            #         name not in pois
            #         and not var.isConstant()
            #         and var.InheritsFrom("RooRealVar")
            #     ):
            #         names.append(os.name)
            # var = it.Next()

            names = _filter_names_by_exclude(names, exclude_expr)
            names.sort()
            return names
        
        # def list_from_workspace(file, workspace, set):
        #     """Create a list of strings from a RooWorkspace set"""
        #     res = []
        #     wsFile = ROOT.TFile(file)
        #     ws = wsFile.Get(workspace)
        #     argSet = ws.set(set)
        #     it = argSet.createIterator()
        #     var = it.Next()
        #     while var:
        #         res.append(var.GetName())
        #         var = it.Next()
        #     return res

        if self.variable in ['','tuto']:
            poiList = ["r"]
        elif self.variable in ["MH"]:
            poiList = ["MH"]
        else:
            poiList = combineVariableDict(self.variable, self.year)['paramStrNoOne']
        
        exclude_expr = ""
        if "combine_impacts" in config:
            exclude_expr = config["combine_impacts"].get("exclude", "")
        paramList = list_nuisance_parameters(datacard_path, "w", "ModelConfig", poiList, exclude_expr=exclude_expr)

        
        branch_map = {i: current_param for i, current_param in enumerate(paramList)}
        return branch_map

    def output(self):        
        current_param = self.branch_data
        
        config = self.get_input_config()
        output_dir = self.get_output_dir()
        outdir = f'outdir_{self.year}_{self.variable}' if self.variable != '' else f'outdir_{self.year}'

        if self.variable == '':
            fitFolderName = f'runFits_mu_fiducial'
        else:
            fitFolderName = f'runFits_{self.variable}'
            
        # output = [os.path.join(output_dir, 'Combine', outdir, fitFolderName)]
        output = [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'impact')]

        output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'impact', f'higgsCombine_paramFit_Test_{current_param}.MultiDimFit.mH{HIGGS_MASS}.root')]
        
        outputFileTargets = []
                
        for _, current_output_path in enumerate(output):
            outputFileTargets.append(law.LocalFileTarget(current_output_path))
            
        # print(outputFileTargets)

        return outputFileTargets

    def run(self):
        current_param = self.branch_data
       
        if self.variable == '':
            fitFolderName = f'runFits_mu_fiducial'
            
        else:
            fitFolderName = f'runFits_{self.variable}'
                  
        config = self.get_input_config()
        output_dir = self.get_output_dir() 
        outdir = f'outdir_{self.year}_{self.variable}' if self.variable != '' else f'outdir_{self.year}'
            
        if self.variable == '':
            datacard_path = os.path.join(output_dir, 'Combine', outdir, f'Datacard_{self.year}.root')
        else:
            datacard_path = os.path.join(output_dir, 'Combine', outdir, f'Datacard_{self.variable}_{self.year}.root')
        
        cwd = os.getcwd()
        if self.batch_flavor == "slurm/psi":
            # Have to use /scratch/batch_username/ for slurm/psi
            if "/work" in output_dir:
                execute_command([f'mkdir -p {output_dir}/Combine/{outdir}/{fitFolderName}/impact'], shell=True)
            else:   
                execute_command([f'xrdfs root://t3dcachedb03.psi.ch:1094/ mkdir -p {output_dir}/Combine/{outdir}/{fitFolderName}/impact'], shell=True)

            os.environ["TARGET_PATH"] = f"/scratch/{os.environ['USER']}/{os.environ['SLURM_JOB_ID']}"
            execute_command([f'mkdir -p $TARGET_PATH/Combine/{outdir}/{fitFolderName}/impact'], shell=True)
            os.chdir(os.path.join(os.environ["TARGET_PATH"], 'Combine', outdir, fitFolderName, 'impact'))
        else:
            execute_command([f'mkdir -p {output_dir}/Combine/{outdir}/{fitFolderName}/impact'], shell=True)
            os.chdir(os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'impact'))

        first_output = os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'asimov')
        if self.variable in ['', 'tuto']:
            pdf_idx_param = "r"
        elif self.variable == "MH":
            pdf_idx_param = "MH"
        else:
            pdf_idx_param = combineVariableDict(self.variable, self.year)['paramStrNoOne'][0]

        def check_pdf_idx(param):
            command = f'root -l -q \'{os.environ["ANALYSIS_PATH"]}/Combine/checkPdfIdx.C("{first_output}/higgsCombinefirstStep_{param}.MultiDimFit.mH{HIGGS_MASS}.root")\''
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            pdfIdx = result.stdout.strip()
            if result.returncode != 0:
                print("Error executing the command:", result.stderr)
                return None
            if pdfIdx.startswith("Processing"):
                pdfIdx = pdfIdx.split('X', 1)[-1]
            pdfIdx = pdfIdx.rstrip(',')
            print(pdfIdx)
            return pdfIdx

        pdfIdx = None
        first_step_path = os.path.join(first_output, f"higgsCombinefirstStep_{pdf_idx_param}.MultiDimFit.mH{HIGGS_MASS}.root")
        if os.path.exists(first_step_path):
            if self.variable != 'MH':
                pdfIdx = check_pdf_idx(pdf_idx_param)
        else:
            print(f"First-step file not found for PDF index extraction: {first_step_path}")

        if self.variable in ['', "tuto"]:
            set_param_string = "r=1"
            if pdfIdx:
                set_param_string = f"{set_param_string},{pdfIdx}"
            arguments = [
                "combine",
                "-M", "MultiDimFit",
                "-d", datacard_path,
                "--algo", "impact",
                "--redefineSignalPOIs", "r",
                "--freezeParameters", "MH",
                "-m", f"{HIGGS_MASS}",
                "-P", f"{current_param}",
                "--floatOtherPOIs", "1",
                "--saveInactivePOI", "1",
                "--robustFit", "1",
                "-n", f"_paramFit_Test_{current_param}",
                "--cminDefaultMinimizerStrategy=0",
                "--cminFallbackAlgo", "Minuit2,Migrad,1:10",
                "--X-rtd", "MINIMIZER_freezeDisassociatedParams",
                "--X-rtd", "MINIMIZER_multiMin_hideConstants",
                "--X-rtd", "MINIMIZER_multiMin_maskConstraints",
                "--X-rtd", "MINIMIZER_multiMin_maskChannels=2",
                "-t", "-1",
                "--setParameters", f"{set_param_string}"
            ]

        elif self.variable in ["MH"]:
            set_param_string = f"MH={HIGGS_MASS}"
            # if pdfIdx:
            #     set_param_string = f"{set_param_string},{pdfIdx}"
            arguments = [
                "combine",
                "-M", "MultiDimFit",
                "-d", datacard_path,
                "--algo", "impact",
                "--redefineSignalPOIs", "MH",
                "--freezeParameters", "r",
                "-m", f"{HIGGS_MASS}",
                "-P", f"{current_param}",
                "--floatOtherPOIs", "1",
                "--saveInactivePOI", "1",
                "--robustFit", "1",
                "-n", f"_paramFit_Test_{current_param}",
                "--cminDefaultMinimizerStrategy=0",
                "--cminFallbackAlgo", "Minuit2,Migrad,1:10",
                "--X-rtd", "MINIMIZER_freezeDisassociatedParams",
                "--X-rtd", "MINIMIZER_multiMin_hideConstants",
                "--X-rtd", "MINIMIZER_multiMin_maskConstraints",
                "--X-rtd", "MINIMIZER_multiMin_maskChannels=2",
                "-t", "-1",
                "--setParameters", f"{set_param_string}"
            ]
            
        else:
            set_param_string = ",".join(combineVariableDict(self.variable, self.year)['paramStr'])
            if pdfIdx:
                set_param_string = f"{set_param_string},{pdfIdx}"
            arguments = [
                "combine",
                "-M", "MultiDimFit",
                "-d", datacard_path,
                "--algo", "impact",
                "--redefineSignalPOIs", f"""{",".join(combineVariableDict(self.variable, self.year)['paramStrNoOne'])}""",
                "--freezeParameters", "MH",
                "-m", f"{HIGGS_MASS}",
                "-P", f"{current_param}",
                "--floatOtherPOIs", "1",
                "--saveInactivePOI", "1",
                "--robustFit", "1",
                "-n", f"_paramFit_Test_{current_param}",
                "--cminDefaultMinimizerStrategy=0",
                "--cminFallbackAlgo", "Minuit2,Migrad,1:10",
                "--X-rtd", "MINIMIZER_freezeDisassociatedParams",
                "--X-rtd", "MINIMIZER_multiMin_hideConstants",
                "--X-rtd", "MINIMIZER_multiMin_maskConstraints",
                "--X-rtd", "MINIMIZER_multiMin_maskChannels=2",
                "-t", "-1",
                "--setParameters", f"{set_param_string}"
            ]


        command = arguments
        # print(command)
        try:
            result = subprocess.run(command, check=True, text=True, capture_output=True)
            print("Script output:", result.stdout)
            print("Script executed successfully.")
        except subprocess.CalledProcessError as e:
            print("Error executing script:", e.stderr)

        # Copy the files back to pnfs if we are on slurm/psi
        if self.batch_flavor == "slurm/psi":
            # Have to copy over the output to the final directory
            # Don't forget to VOMS!
            if "/work" in output_dir:
                slurm_copy_command = [
                    'cp', '-rf',
                    f"{os.environ['TARGET_PATH']}/Combine/",
                    output_dir
                ]
            else:
                slurm_copy_command = [
                    'xrdcp', '-rf',
                    f"{os.environ['TARGET_PATH']}/Combine/",
                    'root://t3dcachedb03.psi.ch:1094//'+output_dir
                ]
            print(slurm_copy_command)
            execute_command(slurm_copy_command)
            # Clean up the temporary directory
            shutil.rmtree(os.environ["TARGET_PATH"])
        
        os.chdir(cwd)
        
class AsimovImpactThirdStep(Task, law.LocalWorkflow, HTCondorWorkflow, SlurmWorkflow): #(law.Task): #(Task, HTCondorWorkflow, law.LocalWorkflow):
    output_dir = law.Parameter(default = '', description="Path to the output directory")
    variable = law.Parameter(default="", description="Variable to be used")
    year = law.Parameter(default='2022', description="Year")

    batch_flavor = law.Parameter(default="htcondor", description="Batch system to use")

    # def requires(self):
    def workflow_requires(self):
        workflow_reqs = super().workflow_requires()

        tasks = {}

        if workflow_reqs:
            tasks.update(workflow_reqs)
                
        #Load central config file
        config = self.get_input_config()
        
        impactConfig = config["combine_impacts"]
        
        if self.output_dir == '':
            output_dir = config['outputFolder']
        else:
            output_dir = self.output_dir
        
        tasks["AsimovImpactSecondStep"] = AsimovImpactSecondStep.req(
            self,
            output_dir=output_dir,  
            workflow=impactConfig["execution"], 
            slurm_partition=impactConfig['batchPartition'],
            slurm_memory=impactConfig['batchMemory'], 
            slurm_max_runtime=impactConfig['batchMaxRuntime'], 
            htcondor_partition=impactConfig['batchPartition'], 
            htcondor_memory=impactConfig['batchMemory'], 
            htcondor_max_runtime=impactConfig['batchMaxRuntime'])
        
        return tasks

    def create_branch_map(self):
        branch_map = {i: current_branch for i, current_branch in enumerate([0])}
        return branch_map

    def output(self):
        config = self.get_input_config()
        output_dir = self.get_output_dir()

        if self.variable == '':
            fitFolderName = f'runFits_mu_fiducial'
        else:
            fitFolderName = f'runFits_{self.variable}'

        outdir = f'outdir_{self.year}_{self.variable}' if self.variable != '' else f'outdir_{self.year}'

        output = []
        if self.variable in ['', 'tuto', 'MH']:
            cat = "r"
            output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'impact', 'impacts')]
            output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'impact', 'impacts', f'impacts.pdf')]
            if self.variable != 'MH':
                output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'impact', 'impacts', 'impacts_corrected_dropBkgModelParams.json')]
            
        else:
            for cat in combineVariableDict(self.variable, self.year)['paramStrNoOne']:
                output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'impact', 'impacts')]
                output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'impact', 'impacts', f'impacts_{cat}.pdf')]
                
        output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'impact', 'impacts', f'impacts.json')]
        
        outputFileTargets = []
                
        for _, current_output_path in enumerate(output):
            outputFileTargets.append(law.LocalFileTarget(current_output_path))
            
        # print(outputFileTargets)

        return outputFileTargets

    def run(self):
        if self.variable == '':
            fitFolderName = f'runFits_mu_fiducial'
            
        else:
            fitFolderName = f'runFits_{self.variable}'
                  
        config = self.get_input_config()
        output_dir = self.get_output_dir() 
        outdir = f'outdir_{self.year}_{self.variable}' if self.variable != '' else f'outdir_{self.year}'
            
        if self.variable == '':
            datacard_path = os.path.join(output_dir, 'Combine', outdir, f'Datacard_{self.year}.root')
        else:
            datacard_path = os.path.join(output_dir, 'Combine', outdir, f'Datacard_{self.variable}_{self.year}.root')

        cwd = os.getcwd()
        if self.batch_flavor == "slurm/psi":
            # Have to use /scratch/batch_username/ for slurm/psi
            if "/work" in output_dir:
                execute_command([f'mkdir -p {output_dir}/Combine/{outdir}/{fitFolderName}/impact/impacts'], shell=True)
            else:   
                execute_command([f'xrdfs root://t3dcachedb03.psi.ch:1094/ mkdir -p {output_dir}/Combine/{outdir}/{fitFolderName}/impact/impacts'], shell=True)

            os.environ["TARGET_PATH"] = f"/scratch/{os.environ['USER']}/{os.environ['SLURM_JOB_ID']}"
            execute_command([f'mkdir -p $TARGET_PATH/Combine/{outdir}/{fitFolderName}/impact/impacts'], shell=True)
            if "/work" in output_dir:
                slurm_copy_command = [
                    'cp', '-rf',
                    f'{output_dir}/Combine/{outdir}/{fitFolderName}/impact',
                    f'{os.environ["TARGET_PATH"]}/Combine/{outdir}/{fitFolderName}'
                ]
            else:
                slurm_copy_command = [
                    'xrdcp', '-rf',
                    'root://t3dcachedb03.psi.ch:1094//'+ f'{output_dir}/Combine/{outdir}/{fitFolderName}/impact',
                    f'{os.environ["TARGET_PATH"]}/Combine/{outdir}/{fitFolderName}'
                ]
            execute_command(slurm_copy_command)
            os.chdir(os.path.join(os.environ["TARGET_PATH"], 'Combine', outdir, fitFolderName, 'impact'))
            temp_output_dir = os.environ["TARGET_PATH"]
        else:
            execute_command([f'mkdir -p {output_dir}/Combine/{outdir}/{fitFolderName}/impact/impacts'], shell=True)
            os.chdir(os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'impact'))
            temp_output_dir = output_dir

        first_output = os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'asimov')
        if self.variable in ['', 'tuto']:
            pdf_idx_param = "r"
            base_param_string = "r=1"
        elif self.variable == "MH":
            pdf_idx_param = "MH"
            base_param_string = f"r=1,MH={HIGGS_MASS}"
        else:
            pdf_idx_param = combineVariableDict(self.variable, self.year)['paramStrNoOne'][0]
            base_param_string = ",".join(combineVariableDict(self.variable, self.year)['paramStr'])

        def check_pdf_idx(param):
            command = f'root -l -q \'{os.environ["ANALYSIS_PATH"]}/Combine/checkPdfIdx.C("{first_output}/higgsCombinefirstStep_{param}.MultiDimFit.mH{HIGGS_MASS}.root")\''
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            pdfIdx = result.stdout.strip()
            if result.returncode != 0:
                print("Error executing the command:", result.stderr)
                return None
            if pdfIdx.startswith("Processing"):
                pdfIdx = pdfIdx.split('X', 1)[-1]
            pdfIdx = pdfIdx.rstrip(',')
            print(pdfIdx)
            return pdfIdx

        pdfIdx = None
        first_step_path = os.path.join(first_output, f"higgsCombinefirstStep_{pdf_idx_param}.MultiDimFit.mH{HIGGS_MASS}.root")
        if os.path.exists(first_step_path):
            pdfIdx = check_pdf_idx(pdf_idx_param)
        else:
            print(f"First-step file not found for PDF index extraction: {first_step_path}")

        set_param_string = None
        if pdfIdx:
            set_param_string = f"{base_param_string},{pdfIdx}"

        if self.variable in ['', 'tuto']:
            exclude_expr = config.get("combine_impacts", {}).get("exclude", "")
            named_params = _list_modelconfig_nuisances(datacard_path, ["r"], exclude_expr=exclude_expr)
            if not named_params:
                raise RuntimeError(f"No nuisance parameters found for impacts in {datacard_path}")
            arguments = [
                "combineTool.py",
                "-M", "Impacts",
                "-d", datacard_path,
                "--freezeParameters", "MH",
                "-m", f"{HIGGS_MASS}",
                "-o", "impacts/impacts.json",
                "--named", ",".join(named_params),
            ]
            if set_param_string is not None:
                arguments.extend(["--setParameters", set_param_string])
            command = arguments
            print('AsimovImpactThirdStep(1): ' + ' '.join(command))
            try:
                result = subprocess.run(command, check=True, text=True, capture_output=True)
                print("Script output:", result.stdout)
                print("Script executed successfully.")
            except subprocess.CalledProcessError as e:
                print("Error executing script:", e.stderr)
                raise

            arguments = [
                "python3",
                f"{os.path.join(os.environ['ANALYSIS_PATH'], 'Plots', 'correctImpacts.py')}",
                "--impactsJson", f"{os.path.join(temp_output_dir, 'Combine', outdir, fitFolderName, 'impact', 'impacts', 'impacts.json')}",
                "--frozenParam", "MH",
                "--dropBkgModelParams"
            ]
            command = arguments
            print('AsimovImpactThirdStep: ' + ' '.join(command))
            try:
                result = subprocess.run(command, check=True, text=True, capture_output=True)
                print("Script output:", result.stdout)
                print("Script executed successfully.")
            except subprocess.CalledProcessError as e:
                print("Error executing script:", e.stderr)
                raise
                
            arguments = [
                "plotImpacts.py",
                "-i", "impacts/impacts_corrected_dropBkgModelParams.json",
                "-o", "impacts/impacts",

            ]
            command = arguments
            print(command)
            try:
                result = subprocess.run(command, check=True, text=True, capture_output=True)
                print("Script output:", result.stdout)
                print("Script executed successfully.")
            except subprocess.CalledProcessError as e:
                print("Error executing script:", e.stderr)

        elif self.variable in ['MH']:
            exclude_expr = config.get("combine_impacts", {}).get("exclude", "")
            named_params = _list_modelconfig_nuisances(datacard_path, ["r"], exclude_expr=exclude_expr)
            if not named_params:
                raise RuntimeError(f"No nuisance parameters found for impacts in {datacard_path}")
            arguments = [
                "combineTool.py",
                "-M", "Impacts",
                "-d", datacard_path,
                "-m", f"{HIGGS_MASS}",
                "--redefineSignalPOIs", "MH",
                "--freezeParameters", "r",
                "-o", "impacts/impacts.json",
                "--named", ",".join(named_params),
            ]
            if set_param_string is not None:
                arguments.extend(["--setParameters", set_param_string])
            command = arguments
            print('AsimovImpactThirdStep(1): ' + ' '.join(command))
            try:
                result = subprocess.run(command, check=True, text=True, capture_output=True)
                print("Script output:", result.stdout)
                print("Script executed successfully.")
            except subprocess.CalledProcessError as e:
                print("Error executing script:", e.stderr)

            
                
            arguments = [
                "plotImpacts.py",
                "-i", "impacts/impacts.json",
                "-o", "impacts/impacts",
                "--POI", "MH"

            ] 
            command = arguments
            print(' '.join(command))
            try:
                result = subprocess.run(command, check=True, text=True, capture_output=True)
                print("Script output:", result.stdout)
                print("Script executed successfully.")
            except subprocess.CalledProcessError as e:
                print("Error executing script:", e.stderr)
                raise
        else:
            arguments = [
                "combineTool.py",
                "-M", "Impacts",
                "-d", datacard_path,
                "--freezeParameters", "MH",
                "-m", f"{HIGGS_MASS}",
                "-o", "impacts/impacts.json",
            ]
            # combineTool.py currently uses "all free parameters" by default (equivalent to --allPars),
            # which pulls in background-model parameters (env_pdf_*) and can crash when old/corrupt
            # param-fit root files are present. Restrict to ModelConfig nuisances by explicitly
            # providing the parameter list via --named.
            exclude_expr = config.get("combine_impacts", {}).get("exclude", "")
            poi_list = combineVariableDict(self.variable, self.year)["paramStrNoOne"]
            named_params = _list_modelconfig_nuisances(datacard_path, poi_list, exclude_expr=exclude_expr)
            if not named_params:
                raise RuntimeError(
                    f"No nuisance parameters found for impacts (variable={self.variable}, year={self.year}). "
                    f"Check workspace ModelConfig nuisances and combine_impacts.exclude='{exclude_expr}'."
                )
            arguments.extend(["--named", ",".join(named_params)])
            if (config["combine_impacts"]["exclude"] != ""):
                arguments.append("--exclude")
                arguments.append(config["combine_impacts"]["exclude"])
            if set_param_string is not None:
                arguments.extend(["--setParameters", set_param_string])
            command = arguments
            # print(command)
            try:
                result = subprocess.run(command, check=True, text=True, capture_output=True)
                print("Script output:", result.stdout)
                print("Script executed successfully.")
            except subprocess.CalledProcessError as e:
                print("Error executing script:", e.stderr)
                
            for cat in combineVariableDict(self.variable, self.year)['paramStrNoOne']:
                arguments = [
                    "plotImpacts.py",
                    "-i", "impacts/impacts.json",
                    "-o", f"impacts/impacts_{cat}",
                    "--POI", f"{cat}"
                ]
                command = arguments
                # print(command)
                try:
                    result = subprocess.run(command, check=True, text=True, capture_output=True)
                    print("Script output:", result.stdout)
                    print("Script executed successfully.")
                except subprocess.CalledProcessError as e:
                    print("Error executing script:", e.stderr)
                    raise

        # Copy the files back to pnfs if we are on slurm/psi
        if self.batch_flavor == "slurm/psi":
            # Have to copy over the output to the final directory
            # Don't forget to VOMS!
            if "/work" in output_dir:
                slurm_copy_command = [
                    'cp', '-rf',
                    f"{os.environ['TARGET_PATH']}/Combine/",
                    output_dir
                ]
            else:
                slurm_copy_command = [
                    'xrdcp', '-rf',
                    f"{os.environ['TARGET_PATH']}/Combine/",
                    'root://t3dcachedb03.psi.ch:1094//'+output_dir
                ]
            print(slurm_copy_command)
            execute_command(slurm_copy_command)
            # Clean up the temporary directory
            shutil.rmtree(os.environ["TARGET_PATH"])
            
        os.chdir(cwd)


class AsimovCovCorrHesse(Task, law.LocalWorkflow, HTCondorWorkflow, SlurmWorkflow): #(law.Task): #(Task, HTCondorWorkflow, law.LocalWorkflow):
    output_dir = law.Parameter(default = '', description="Path to the output directory")
    variable = law.Parameter(default="", description="Variable to be used")
    year = law.Parameter(default='2022', description="Year")

    batch_flavor = law.Parameter(default="htcondor", description="Batch system to use")

    # def requires(self):
    def workflow_requires(self):
        workflow_reqs = super().workflow_requires()

        tasks = {}

        if workflow_reqs:
            tasks.update(workflow_reqs)
        
        if self.variable == '':
            print("Running AsimovCovCorrHesse for inclusive does not make sense. Please specify a variable.")
            exit(1)
        
        config = self.get_input_config()
        output_dir = self.get_output_dir()
        
        hesseConfig = config["combine_hesse"]
        r
        
        tasks["RunT2WS"] = RunText2Workspace(output_dir=output_dir, variable=self.variable, year=self.year, batch_flavor=self.batch_flavor, workflow=hesseConfig["execution"], version=self.variable, slurm_partition=hesseConfig['batchPartition'], slurm_memory=hesseConfig['batchMemory'], slurm_max_runtime=hesseConfig['batchMaxRuntime'], htcondor_partition=hesseConfig['batchPartition'], htcondor_memory=hesseConfig['batchMemory'], htcondor_max_runtime=hesseConfig['batchMaxRuntime'])
        
        return tasks

    def create_branch_map(self):
        branch_map = {i: current_branch for i, current_branch in enumerate([0])}
        return branch_map

    def output(self):        
        if self.variable == '':
            print("Running AsimovCovCorrHesse for inclusive does not make sense. Please specify a variable.")
            exit(1)
        
        config = self.get_input_config()
        output_dir = self.get_output_dir()

        if self.variable == '':
            fitFolderName = f'runFits_mu_fiducial'
        else:
            fitFolderName = f'runFits_{self.variable}'
            
        if self.variable == "":
            output = []
        else:
            # output = [os.path.join(output_dir, 'Combine', outdir, fitFolderName)]
            output = [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'hesse')]
            
            
            output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'hesse', f'robustHessefirstStep.root')]
            output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'hesse', f'multidimfitfirstStep.root')]
            output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'hesse', f'higgsCombinefirstStep.MultiDimFit.mH{HIGGS_MASS}.root')]
        
        outputFileTargets = []
                
        for _, current_output_path in enumerate(output):
            outputFileTargets.append(law.LocalFileTarget(current_output_path))
            
        # print(outputFileTargets)

        return outputFileTargets

    def run(self):
       
        if self.variable == '':
            print("Running AsimovCovCorrHesse for inclusive does not make sense. Please specify a variable.")
            exit(1)
            
        else:
            fitFolderName = f'runFits_{self.variable}'
                  
        config = self.get_input_config()
        output_dir = self.get_output_dir() 
            
        if self.variable == '':
            datacard_path = os.path.join(output_dir, 'Combine', outdir, f'Datacard_{self.year}.root')
        else:
            datacard_path = os.path.join(output_dir, 'Combine', outdir, f'Datacard_{self.variable}_{self.year}.root')
        
        cwd = os.getcwd()
        
        if self.batch_flavor == "slurm/psi":
            # Have to use /scratch/batch_username/ for slurm/psi
            if "/work" in output_dir:
                execute_command([f'mkdir -p {output_dir}/Combine/{fitFolderName}/hesse'], shell=True)
            else:   
                execute_command([f'xrdfs root://t3dcachedb03.psi.ch:1094/ mkdir -p {output_dir}/Combine/{fitFolderName}/hesse'], shell=True)

            os.environ["TARGET_PATH"] = f"/scratch/{os.environ['USER']}/{os.environ['SLURM_JOB_ID']}"
            execute_command([f'mkdir -p $TARGET_PATH/Combine/{fitFolderName}/hesse'], shell=True)
            os.chdir(os.path.join(os.environ["TARGET_PATH"], 'Combine', fitFolderName, 'hesse'))
        else:
            execute_command([f'mkdir -p {output_dir}/Combine/{fitFolderName}/hesse'], shell=True)
            os.chdir(os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'hesse'))

        # Determine which discrete pdfindex categories actually exist in the workspace.
        # This is important when some bins use merged categories (e.g. *_catMerged_*)
        # and therefore do not define the usual *_cat0/cat1/cat2_* RooCategories.
        pdf_indices = combineVariableDict(self.variable, self.year)['pdfIndeces']
        cache_key = (datacard_path,)
        if cache_key not in _PDFINDEX_CACHE and os.path.exists(datacard_path):
            datacard_pdf_indices = extract_pdf_indices(datacard_path)
            if datacard_pdf_indices:
                _PDFINDEX_CACHE[cache_key] = datacard_pdf_indices
        if cache_key in _PDFINDEX_CACHE:
            pdf_indices = _PDFINDEX_CACHE[cache_key]
        arguments = [
            "combine",
            "-M", "MultiDimFit",
            datacard_path,
            "--freezeParameters", "MH",
            "-m", f"{HIGGS_MASS}",
            "-n", "firstStep",
            "--saveWorkspace",
            "--saveFitResult",
            "--floatOtherPOIs", "1",
            "--robustHesse", "1",
            "--robustHesseSave", "1",
            "--cminDefaultMinimizerStrategy=0",
            "--cminFallbackAlgo", "Minuit2,Migrad,1:10",
            "--X-rtd", "MINIMIZER_freezeDisassociatedParams",
            "--X-rtd", "MINIMIZER_multiMin_hideConstants",
            "--X-rtd", "MINIMIZER_multiMin_maskConstraints",
            "--X-rtd", "MINIMIZER_multiMin_maskChannels=2",
            "--X-rtd", "MINIMIZER_skipDiscreteIterations",
            "-t", "-1",
            "--setParameters", f"""{",".join(combineVariableDict(self.variable, self.year)['paramStr'])}"""
        ]
        arguments.extend(_save_specified_index_args(pdf_indices))
        command = arguments
        # print(command)
        try:
            result = subprocess.run(command, check=True, text=True, capture_output=True)
            print("Script output:", result.stdout)
            print("Script executed successfully.")
        except subprocess.CalledProcessError as e:
            print("Error executing script:", e.stderr)

        # Copy the files back to pnfs if we are on slurm/psi
        if self.batch_flavor == "slurm/psi":
            # Have to copy over the output to the final directory
            # Don't forget to VOMS!
            if "/work" in output_dir:
                slurm_copy_command = [
                    'cp', '-rf',
                    f"{os.environ['TARGET_PATH']}/Combine/",
                    output_dir
                ]
            else:
                slurm_copy_command = [
                    'xrdcp', '-rf',
                    f"{os.environ['TARGET_PATH']}/Combine/",
                    'root://t3dcachedb03.psi.ch:1094//'+output_dir
                ]
            print(slurm_copy_command)
            execute_command(slurm_copy_command)
            # Clean up the temporary directory
            shutil.rmtree(os.environ["TARGET_PATH"])
            
        os.chdir(cwd)
        
class AsimovCovCorr(Task, law.LocalWorkflow, HTCondorWorkflow, SlurmWorkflow): #(law.Task): #(Task, HTCondorWorkflow, law.LocalWorkflow):
    output_dir = law.Parameter(default = '', description="Path to the output directory")
    variable = law.Parameter(default="", description="Variable to be used")
    year = law.Parameter(default='2022', description="Year")
    noPreliminary = law.Parameter(default=False, description="Flag, if final plot should bear the Preliminary.")

    batch_flavor = law.Parameter(default="htcondor", description="Batch system to use")


    # def requires(self):
    def workflow_requires(self):
        workflow_reqs = super().workflow_requires()

        tasks = {}

        if workflow_reqs:
            tasks.update(workflow_reqs)
        
        config = self.get_input_config()
        output_dir = self.get_output_dir()
        
        if self.variable == '':
            print("Running AsimovCovCorr for inclusive does not make sense. Please specify a variable.")
            exit(1)
        else:
            version = self.variable
            tasks["AsimovCovCorrHesse"] = AsimovCovCorrHesse(output_dir=output_dir, variable=self.variable, year=self.year, version=version, workflow=config["combine_hesse"]["execution"], batch_flavor=self.batch_flavor, slurm_partition=config["combine_hesse"]['batchPartition'], slurm_memory=config["combine_hesse"]['batchMemory'], slurm_max_runtime=config["combine_hesse"]['batchMaxRuntime'], htcondor_partition=config["combine_hesse"]['batchPartition'], htcondor_memory=config["combine_hesse"]['batchMemory'], htcondor_max_runtime=config["combine_hesse"]['batchMaxRuntime'])
        
        return tasks

    def create_branch_map(self):
        branch_map = {i: current_branch for i, current_branch in enumerate([0])}
        return branch_map

    def output(self):        
        if self.variable == '':
            print("Running AsimovCovCorr for inclusive does not make sense. Please specify a variable.")
            exit(1)
        
        config = self.get_input_config()
        output_dir = self.get_output_dir()

        if self.variable == '':
            fitFolderName = f'runFits_mu_fiducial'
        else:
            fitFolderName = f'runFits_{self.variable}'
            
        # output = [os.path.join(output_dir, 'Combine', outdir, fitFolderName)]
        output = [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'hesse', 'Plots')]

        output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'hesse', 'Plots', f'corrMatrix_{self.variable}_syst.pdf')]
        output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'hesse', 'Plots', f'corrMatrix_{self.variable}_syst.png')]
        output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'hesse', 'Plots', f'covMatrix_{self.variable}_syst.pdf')]
        output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'hesse', 'Plots', f'covMatrix_{self.variable}_syst.png')]
        
        outputFileTargets = []
                
        for _, current_output_path in enumerate(output):
            outputFileTargets.append(law.LocalFileTarget(current_output_path))
            
        # print(outputFileTargets)

        return outputFileTargets

    def run(self):
       
        if self.variable == '':
            # Does not make sense inclusively
            return True
            
        else:
            fitFolderName = f'runFits_{self.variable}'
                  
        config = self.get_input_config()
        output_dir = self.get_output_dir() 
            
        # if self.variable == '':
        #     datacard_path = os.path.join(output_dir, 'Combine', outdir, f'Datacard_{self.year}.root')
        # else:
        #     datacard_path = os.path.join(output_dir, 'Combine', outdir, f'Datacard_{self.variable}_{self.year}.root')
        
        if self.batch_flavor == "slurm/psi":
            # Have to use /scratch/batch_username/ for slurm/psi
            if "/work" in output_dir:
                execute_command([f'mkdir -p {output_dir}/Combine/{fitFolderName}/hesse/Plots'], shell=True)
            else:   
                execute_command([f'xrdfs root://t3dcachedb03.psi.ch:1094/ mkdir -p {output_dir}/Combine/{fitFolderName}/hesse/Plots'], shell=True)

            os.environ["TARGET_PATH"] = f"/scratch/{os.environ['USER']}/{os.environ['SLURM_JOB_ID']}"
            execute_command([f'mkdir -p $TARGET_PATH/Combine/{fitFolderName}/hesse/Plots'], shell=True)

            if "/work" in output_dir:
                slurm_copy_command = [
                    'cp', '-rf',
                    f'{output_dir}/Combine/{fitFolderName}/hesse',
                    f'{os.environ["TARGET_PATH"]}/Combine/{fitFolderName}/'
                ]
            else:
                slurm_copy_command = [
                    'xrdcp', '-rf',
                    'root://t3dcachedb03.psi.ch:1094//'+ f'{output_dir}/Combine/{fitFolderName}/hesse',
                    f'{os.environ["TARGET_PATH"]}/Combine/{fitFolderName}/'
                ]
            execute_command(slurm_copy_command)
            temp_output_dir = os.environ["TARGET_PATH"]
            # output_dir = os.path.join(os.environ["TARGET_PATH"], 'Combine', fitFolderName, 'hesse', 'Plots')
        else:
            execute_command([f'mkdir -p {output_dir}/Combine/{fitFolderName}/hesse/Plots'], shell=True)
            temp_output_dir = output_dir
            # output_dir = os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'hesse', 'Plots')            
        
        cwd = os.getcwd()
        os.chdir(os.path.join(os.environ["ANALYSIS_PATH"], 'Plots'))

        arguments = [
            "python3",
            f"{os.path.join(os.environ['ANALYSIS_PATH'], 'Plots', 'makeCorrMatrix.py')}",
            "--inputJson", f"{os.path.join(os.environ['ANALYSIS_PATH'], 'Plots', 'inputs_robustHesse.json')}",
            "--mode", f"{self.variable}",
            "--input", f"{os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'hesse', f'robustHessefirstStep.root')}",
            "--output", f"{os.path.join(temp_output_dir, 'Combine', outdir, fitFolderName, 'hesse', 'Plots')}",
            "--translate", f"{os.path.join(os.environ['ANALYSIS_PATH'], 'Plots', 'poi_differential_hesse_noLabels.json')}"
        ]
        if convert_boolean_string(self.noPreliminary):
            arguments.append("--noPreliminary")
        command = arguments
        print(command)
        try:
            result = subprocess.run(command, check=True, text=True, capture_output=True)
            print("Script output:", result.stdout)
            print("Script executed successfully.")
        except subprocess.CalledProcessError as e:
            print("Error executing script:", e.stderr)
            
        arguments = [
            "python3",
            f"{os.path.join(os.environ['ANALYSIS_PATH'], 'Plots', 'makeCorrMatrix.py')}",
            "--inputJson", f"{os.path.join(os.environ['ANALYSIS_PATH'], 'Plots', 'inputs_robustHesse.json')}",
            "--mode", f"{self.variable}",
            "--input", f"{os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'hesse', f'robustHessefirstStep.root')}",
            "--output", f"{os.path.join(temp_output_dir, 'Combine', outdir, fitFolderName, 'hesse', 'Plots')}",
            "--translate", f"{os.path.join(os.environ['ANALYSIS_PATH'], 'Plots', 'poi_differential_hesse_noLabels.json')}",
            "--doCov"
        ]
        if convert_boolean_string(self.noPreliminary):
            arguments.append("--noPreliminary")
        command = arguments
        print(command)
        try:
            result = subprocess.run(command, check=True, text=True, capture_output=True)
            print("Script output:", result.stdout)
            print("Script executed successfully.")
        except subprocess.CalledProcessError as e:
            print("Error executing script:", e.stderr)
        
        # Copy the files back to pnfs if we are on slurm/psi
        if self.batch_flavor == "slurm/psi":
            # Have to copy over the output to the final directory
            # Don't forget to VOMS!
            if "/work" in output_dir:
                slurm_copy_command = [
                    'cp', '-rf',
                    f"{os.environ['TARGET_PATH']}/Combine/",
                    output_dir
                ]
            else:
                slurm_copy_command = [
                    'xrdcp', '-rf',
                    f"{os.environ['TARGET_PATH']}/Combine/",
                    'root://t3dcachedb03.psi.ch:1094//'+output_dir
                ]
            print(slurm_copy_command)
            execute_command(slurm_copy_command)
            # Clean up the temporary directory
            shutil.rmtree(os.environ["TARGET_PATH"])
            
        os.chdir(cwd)

class UnblindedFitSystSingle(Task, law.LocalWorkflow, HTCondorWorkflow, SlurmWorkflow): #(law.Task): #(Task, HTCondorWorkflow, law.LocalWorkflow):
    output_dir = law.Parameter(default = '', description="Path to the output directory")
    variable = law.Parameter(default="", description="Variable to be used")
    year = law.Parameter(default='2022', description="Year")

    batch_flavor = law.Parameter(default="htcondor", description="Batch system to use")

    # def requires(self):
    def workflow_requires(self):
        workflow_reqs = super().workflow_requires()

        tasks = {}

        if workflow_reqs:
            tasks.update(workflow_reqs)
        
        config = self.get_input_config()
        output_dir = self.get_output_dir()

        fitConfig = config["combine_fit"] 
 
        tasks["RunT2WS"] = RunText2Workspace(output_dir=output_dir, variable=self.variable, year=self.year, version=self.variable if self.variable != "" else "inclusive", workflow=fitConfig["execution"], batch_flavor=self.batch_flavor, slurm_partition=fitConfig['batchPartition'], slurm_memory=fitConfig['batchMemory'], slurm_max_runtime=fitConfig['batchMaxRuntime'], htcondor_partition=fitConfig['batchPartition'], htcondor_memory=fitConfig['batchMemory'], htcondor_max_runtime=fitConfig['batchMaxRuntime'])        

        return tasks
    
    def create_branch_map(self):
        if self.variable == '':
            param_list = ["r"]
        else:
            param_list = combineVariableDict(self.variable, self.year)['paramStrNoOne']
        branch_map = {i: current_cat for i, current_cat in enumerate(param_list)}
        return branch_map

    def output(self):
        current_cat = self.branch_data
        
        config = self.get_input_config()
        output_dir = self.get_output_dir()

        if self.variable == '':
            fitFolderName = f'runFits_mu_fiducial'
        else:
            fitFolderName = f'runFits_{self.variable}'
        
        output = [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'dataFit')]
        output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'dataFit', f'higgsCombineDataPostFitBestFit_{current_cat}.MultiDimFit.mH{HIGGS_MASS}.root')]
        
        outputFileTargets = []
                
        for _, current_output_path in enumerate(output):
            outputFileTargets.append(law.LocalFileTarget(current_output_path))
            
        # print("AsimovFitCategorySyst", outputFileTargets)

        return outputFileTargets

    def run(self):
        current_cat = self.branch_data
        
        if self.variable == '':
            fitFolderName = f'runFits_mu_fiducial'
            
        else:
            fitFolderName = f'runFits_{self.variable}'
                    
        #Load central config file
        with open(configYamlPath, 'r') as file:
            config = yaml.safe_load(file)

        # default values, if not set in config
        cminApproxPreFitTolerance = config.get("combine_fit", {}).get("cminApproxPreFitTolerance", 0.01)
        rMin = config.get("combine_fit", {}).get("rMin", 0.7)
        rMax = config.get("combine_fit", {}).get("rMax", 1.6)

        if self.output_dir == '':
            output_dir = config['outputFolder']
        else:
            output_dir = self.output_dir
        
        
        cwd = os.getcwd()
        
        if self.variable == '':
            datacard_path = os.path.join(output_dir, 'Combine', outdir, f'Datacard_{self.year}.root')
        else:
            datacard_path = os.path.join(output_dir, 'Combine', outdir, f'Datacard_{self.variable}_{self.year}.root')
        if self.batch_flavor == "slurm/psi":
            # Have to use /scratch/batch_username/ for slurm/psi
            os.environ["TARGET_PATH"] = f"/scratch/{os.environ['USER']}/{os.environ['SLURM_JOB_ID']}"
            execute_command([f'mkdir -p $TARGET_PATH/Combine/{fitFolderName}/dataFit'], shell=True)
            os.chdir(os.path.join(os.environ["TARGET_PATH"], 'Combine', fitFolderName, 'dataFit'))
        else:
            execute_command([f'mkdir -p {output_dir}/Combine/{fitFolderName}/dataFit'], shell=True)
            os.chdir(os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'dataFit'))

        if self.variable == '':
            arguments = [
                "combine",
                "-M", "MultiDimFit",
                "-d", datacard_path,
                "--freezeParameters", "MH",
                "-m", f"{HIGGS_MASS}",
                "-n", f"DataPostFitBestFit_{current_cat}",
                "--cminDefaultMinimizerStrategy=0",
                "--cminFallbackAlgo", "Minuit2,Migrad,1:10",
                "--algo", "singles",
                "--X-rtd", "MINIMIZER_freezeDisassociatedParams",
                "--X-rtd", "MINIMIZER_multiMin_hideConstants",
                "--X-rtd", "MINIMIZER_multiMin_maskConstraints",
                "--X-rtd", "MINIMIZER_multiMin_maskChannels=2",
                "-P", "r",
                "--floatOtherPOIs", "1",
                "--saveWorkspace",
                "--saveFitResult",
                "--cminApproxPreFitTolerance", f"{cminApproxPreFitTolerance}",
                "--rMin", f"{rMin}",
                "--rMax", f"{rMax}",
            ]
            command = arguments
            # print(command)
            try:
                result = subprocess.run(command, check=True, text=True, capture_output=True)
                print("Script output:", result.stdout)
                print("Script executed successfully.")
            except subprocess.CalledProcessError as e:
                print("Error executing script:", e.stderr)
            
        else:         
            arguments = [
                "combine",
                "-M", "MultiDimFit",
                datacard_path,
                "--freezeParameters", "MH",
                "-m", f"{HIGGS_MASS}",
                "-n", f"DataPostFitBestFit_{current_cat}",
                "--cminDefaultMinimizerStrategy=0",
                "--cminFallbackAlgo", "Minuit2,Migrad,1:10",
                "--algo", "singles",
                "--X-rtd", "MINIMIZER_freezeDisassociatedParams",
                "--X-rtd", "MINIMIZER_multiMin_hideConstants",
                "--X-rtd", "MINIMIZER_multiMin_maskConstraints",
                "--X-rtd", "MINIMIZER_multiMin_maskChannels=2",
                "-P", f"{current_cat}",
                "--cminApproxPreFitTolerance", f"{cminApproxPreFitTolerance}",
                # "--stepSize", "0.05", 
                # "--setCrossingTolerance", "0.00005",
                "--saveFitResult",
                "--floatOtherPOIs", "1",
                "--saveWorkspace",
                "--saveSpecifiedIndex", f"""{",".join(combineVariableDict(self.variable, self.year)['pdfIndeces'])}""",
            ]
            command = arguments
            # print(command)
            try:
                result = subprocess.run(command, check=True, text=True, capture_output=True)
                print("Script output:", result.stdout)
                print("Script executed successfully.")
            except subprocess.CalledProcessError as e:
                print("Error executing script:", e.stderr)

        # Copy the files back to pnfs if we are on slurm/psi
        if self.batch_flavor == "slurm/psi":
            # Have to copy over the output to the final directory
            # Don't forget to VOMS!
            slurm_copy_command = [
                'xrdcp', '-rf',
                f"{os.environ['TARGET_PATH']}/Combine/",
                'root://t3dcachedb03.psi.ch:1094//'+output_dir
            ]
            print(slurm_copy_command)
            execute_command(slurm_copy_command)
            # Clean up the temporary directory
            shutil.rmtree(os.environ["TARGET_PATH"])

        os.chdir(cwd)

class UnblindedFitStatSingle(Task, law.LocalWorkflow, HTCondorWorkflow, SlurmWorkflow): #(law.Task): #(Task, HTCondorWorkflow, law.LocalWorkflow):
    output_dir = law.Parameter(default = '', description="Path to the output directory")
    variable = law.Parameter(default="", description="Variable to be used")
    year = law.Parameter(default='2022', description="Year")
    
    batch_flavor = law.Parameter(default="htcondor", description="Batch system to use")

    # def requires(self):
    def workflow_requires(self):
        workflow_reqs = super().workflow_requires()

        tasks = {}

        if workflow_reqs:
            tasks.update(workflow_reqs)
        
        config = self.get_input_config()
        output_dir = self.get_output_dir()

        fitConfig = config["combine_fit"]        

        tasks["RunT2WS"] = RunText2Workspace(output_dir=output_dir, variable=self.variable, year=self.year, version=self.variable if self.variable != "" else "inclusive", workflow=fitConfig["execution"], batch_flavor=self.batch_flavor, slurm_partition=fitConfig['batchPartition'], slurm_memory=fitConfig['batchMemory'], slurm_max_runtime=fitConfig['batchMaxRuntime'], htcondor_partition=fitConfig['batchPartition'], htcondor_memory=fitConfig['batchMemory'], htcondor_max_runtime=fitConfig['batchMaxRuntime'])        
        tasks["UnblindedFitSystSingle"] = UnblindedFitSystSingle(output_dir=output_dir, variable=self.variable, year=self.year, batch_flavor=self.batch_flavor, version=self.variable if self.variable != "" else "inclusive", workflow=fitConfig["execution"], slurm_partition=fitConfig['batchPartition'], slurm_memory=fitConfig['batchMemory'], slurm_max_runtime=fitConfig['batchMaxRuntime'], htcondor_partition=fitConfig['batchPartition'], htcondor_memory=fitConfig['batchMemory'], htcondor_max_runtime=fitConfig['batchMaxRuntime'])

        return tasks
    
    def create_branch_map(self):
        if self.variable == '':
            param_list = ["r"]
        else:
            param_list = combineVariableDict(self.variable, self.year)['paramStrNoOne']
        branch_map = {i: current_cat for i, current_cat in enumerate(param_list)}
        return branch_map

    def output(self):
        cat = self.branch_data
        
        config = self.get_input_config()
        output_dir = self.get_output_dir()

        if self.variable == '':
            fitFolderName = f'runFits_mu_fiducial'
        else:
            fitFolderName = f'runFits_{self.variable}'
            
        output = []

        output = [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'dataFit', f'higgsCombineDataPostFitBestFitStat_{cat}.MultiDimFit.mH{HIGGS_MASS}.root')]
        
        outputFileTargets = []
                
        for _, current_output_path in enumerate(output):
            outputFileTargets.append(law.LocalFileTarget(current_output_path))
                    
        return outputFileTargets

    def run(self):
        cat = self.branch_data

        if self.variable == '':
            fitFolderName = f'runFits_mu_fiducial'
            
        else:
            fitFolderName = f'runFits_{self.variable}'
                    
        #Load central config file
        with open(configYamlPath, 'r') as file:
            config = yaml.safe_load(file)

        # default values, if not set in config
        cminApproxPreFitTolerance = config.get("combine_fit", {}).get("cminApproxPreFitTolerance", 0.01)
        rMin = config.get("combine_fit", {}).get("rMin", 0.7)
        rMax = config.get("combine_fit", {}).get("rMax", 1.6)

        if self.output_dir == '':
            output_dir = config['outputFolder']
        else:
            output_dir = self.output_dir  

        cwd = os.getcwd()
        if self.batch_flavor == "slurm/psi":
            # Have to use /scratch/batch_username/ for slurm/psi
            if "/work" in output_dir:
                execute_command([f'mkdir -p {output_dir}/Combine/{fitFolderName}/dataFit'], shell=True)
            else:   
                execute_command([f'xrdfs root://t3dcachedb03.psi.ch:1094/ mkdir -p {output_dir}/Combine/{fitFolderName}/dataFit'], shell=True)

            os.environ["TARGET_PATH"] = f"/scratch/{os.environ['USER']}/{os.environ['SLURM_JOB_ID']}"
            execute_command([f'mkdir -p $TARGET_PATH/Combine/{fitFolderName}/dataFit'], shell=True)
            os.chdir(os.path.join(os.environ["TARGET_PATH"], 'Combine', fitFolderName, 'dataFit'))
        else:
            execute_command([f'mkdir -p {output_dir}/Combine/{fitFolderName}/dataFit'], shell=True)
            os.chdir(os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'dataFit'))
    
        firstStepPath = os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'dataFit', f"higgsCombineDataPostFitBestFit_{cat}.MultiDimFit.mH{HIGGS_MASS}.root")
        local_first_step = os.path.join(os.getcwd(), os.path.basename(firstStepPath))
        if not os.path.exists(firstStepPath):
            raise RuntimeError(f"Required best fit snapshot not found: {firstStepPath}")
        if os.path.abspath(local_first_step) != os.path.abspath(firstStepPath):
            shutil.copy2(firstStepPath, local_first_step)
        input_path = local_first_step if os.path.exists(local_first_step) else firstStepPath
                
        if self.variable == '':
            arguments = [
                "combine",
                "-M", "MultiDimFit",
                input_path,
                "--freezeParameters", "MH",
                "-m", f"{HIGGS_MASS}",
                "-n", f"DataPostFitBestFitStat_{cat}",
                "--cminDefaultMinimizerStrategy=0",
                "--cminFallbackAlgo", "Minuit2,Migrad,1:10",
                "--algo", "singles",
                "--rMin", f"{rMin}",
                "--rMax", f"{rMax}",
                "--X-rtd", "MINIMIZER_freezeDisassociatedParams",
                "--X-rtd", "MINIMIZER_multiMin_hideConstants",
                "--X-rtd", "MINIMIZER_multiMin_maskConstraints",
                "--X-rtd", "MINIMIZER_multiMin_maskChannels=2",
                "-P", f"{cat}",
                "--floatOtherPOIs", "1",
                "--saveWorkspace",
                "--saveFitResult",
                "--snapshotName", "MultiDimFit",
                "-w", "w",
                "--cminApproxPreFitTolerance", f"{cminApproxPreFitTolerance}",
            ]
            command = arguments
            # print(command)
            try:
                result = subprocess.run(command, check=True, text=True, capture_output=True)
                print("Script output:", result.stdout)
                print("Script executed successfully.")
            except subprocess.CalledProcessError as e:
                print("Error executing script:", e.stderr)
            
        else:
            arguments = [
                "combineTool.py",
                "-M", "MultiDimFit",
                input_path,
                "--freezeParameters", "MH",
                "-m", f"{HIGGS_MASS}",
                "-n", f"DataPostFitBestFitStat_{cat}",
                "--cminDefaultMinimizerStrategy=0",
                "--cminFallbackAlgo", "Minuit2,Migrad,1:10",
                "--algo", "singles",
                "--X-rtd", "MINIMIZER_freezeDisassociatedParams",
                "--X-rtd", "MINIMIZER_multiMin_hideConstants",
                "--X-rtd", "MINIMIZER_multiMin_maskConstraints",
                "--X-rtd", "MINIMIZER_multiMin_maskChannels=2",
                "-P", f"{cat}",
                "--saveFitResult",
                "--floatOtherPOIs", "1",
                "--saveWorkspace",
                "--snapshotName", "MultiDimFit",
                "-w", "w",
            ]
            command = arguments
            # print(command)
            try:
                result = subprocess.run(command, check=True, text=True, capture_output=True)
                print("Script output:", result.stdout)
                print("Script executed successfully.")
            except subprocess.CalledProcessError as e:
                print("Error executing script:", e.stderr)
            
        # Copy the files back to pnfs if we are on slurm/psi
        if self.batch_flavor == "slurm/psi":
            # Have to copy over the output to the final directory
            # Don't forget to VOMS!
            if "/work" in output_dir:
                slurm_copy_command = [
                    'cp', '-rf',
                    f"{os.environ['TARGET_PATH']}/Combine/",
                    output_dir
                ]
            else:
                slurm_copy_command = [
                    'xrdcp', '-rf',
                    f"{os.environ['TARGET_PATH']}/Combine/",
                    'root://t3dcachedb03.psi.ch:1094//'+output_dir
                ]
            print(slurm_copy_command)
            execute_command(slurm_copy_command)
            # Clean up the temporary directory
            shutil.rmtree(os.environ["TARGET_PATH"])
                
        
        os.chdir(cwd)
        
class UnblindedFitCategorySyst(Task, law.LocalWorkflow, HTCondorWorkflow, SlurmWorkflow): #(law.Task): #(Task, HTCondorWorkflow, law.LocalWorkflow):
    output_dir = law.Parameter(default = '', description="Path to the output directory")
    variable = law.Parameter(default="", description="Variable to be used")
    year = law.Parameter(default='2022', description="Year")
    nPoints = law.Parameter(default=30, description="Number of points for the LL scan")
    
    batch_flavor = law.Parameter(default="htcondor", description="Batch system to use")

    # def requires(self):
    def workflow_requires(self):
        workflow_reqs = super().workflow_requires()

        tasks = {}

        if workflow_reqs:
            tasks.update(workflow_reqs)
        
        config = self.get_input_config()
        output_dir = self.get_output_dir()
               
        fitConfig = config["combine_fit"]

        tasks["UnblindedFitSystSingle"] = UnblindedFitSystSingle(output_dir=output_dir, variable=self.variable, year=self.year, batch_flavor=self.batch_flavor, version=self.variable if self.variable != "" else "inclusive", workflow=fitConfig["execution"], slurm_partition=fitConfig['batchPartition'], slurm_memory=fitConfig['batchMemory'], slurm_max_runtime=fitConfig['batchMaxRuntime'], htcondor_partition=fitConfig['batchPartition'], htcondor_memory=fitConfig['batchMemory'], htcondor_max_runtime=fitConfig['batchMaxRuntime'])
        
        return tasks
    
    def create_branch_map(self):
        if self.variable == '':
            cats = ["r"]
        else:
            cats = combineVariableDict(self.variable, self.year)['paramStrNoOne']
        branch_data = [(cat, point) for cat in cats for point in range(int(self.nPoints))]
        branch_map = {i: current_branch for i, current_branch in enumerate(branch_data)}
        return branch_map

    def output(self):
        current_cat, current_point = self.branch_data
        
        config = self.get_input_config()
        output_dir = self.get_output_dir()

        if self.variable == '':
            fitFolderName = f'runFits_mu_fiducial'
        else:
            fitFolderName = f'runFits_{self.variable}'
        
        output = [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'dataFit')]
        output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'dataFit', f'higgsCombineDataPostFitScanFit_{current_cat}.POINTS.{current_point}.{current_point}.MultiDimFit.mH{HIGGS_MASS}.root')]
        
        outputFileTargets = []
                
        for _, current_output_path in enumerate(output):
            outputFileTargets.append(law.LocalFileTarget(current_output_path))
            
        # print("AsimovFitCategorySyst", outputFileTargets)

        return outputFileTargets

    def run(self):
        current_cat, current_point = self.branch_data
        
        if self.variable == '':
            fitFolderName = f'runFits_mu_fiducial'
            
        else:
            fitFolderName = f'runFits_{self.variable}'
                    
        config = self.get_input_config()
        output_dir = self.get_output_dir() 
        outdir = f"outdir_{self.variable}_{self.year}" if self.variable != "" else f"outdir_{self.year}"

        # default values, if not set in config
        cminApproxPreFitTolerance = config.get("combine_fit", {}).get("cminApproxPreFitTolerance", 0.01)
        rMin = config.get("combine_fit", {}).get("rMin", 0.7)
        rMax = config.get("combine_fit", {}).get("rMax", 1.6)

         
        
        cwd = os.getcwd()
        if self.batch_flavor == "slurm/psi":
            # Have to use /scratch/batch_username/ for slurm/psi
            if "/work" in output_dir:
                execute_command([f'mkdir -p {output_dir}/Combine/{fitFolderName}/dataFit'], shell=True)
            else:   
                execute_command([f'xrdfs root://t3dcachedb03.psi.ch:1094/ mkdir -p {output_dir}/Combine/{fitFolderName}/dataFit'], shell=True)

            os.environ["TARGET_PATH"] = f"/scratch/{os.environ['USER']}/{os.environ['SLURM_JOB_ID']}"
            execute_command([f'mkdir -p $TARGET_PATH/Combine/{fitFolderName}/dataFit'], shell=True)
            os.chdir(os.path.join(os.environ["TARGET_PATH"], 'Combine', fitFolderName, 'dataFit'))
        else:
            execute_command([f'mkdir -p {output_dir}/Combine/{fitFolderName}/dataFit'], shell=True)
            os.chdir(os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'dataFit'))
        
        firstStepPath = os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'dataFit', f"higgsCombineDataPostFitBestFit_{current_cat}.MultiDimFit.mH{HIGGS_MASS}.root")
        n_points = int(self.nPoints)
            
        if self.variable == '':
            arguments = [
                "combine",
                "-M", "MultiDimFit",
                "-d", firstStepPath,
                "--freezeParameters", "MH",
                "-m", f"{HIGGS_MASS}",
                "-n", f"DataPostFitScanFit_{current_cat}.POINTS.{current_point}.{current_point}",
                "--cminDefaultMinimizerStrategy=0",
                "--cminFallbackAlgo", "Minuit2,Migrad,1:10",
                "--algo", "grid",
                "--points", f"{n_points}",
                "--rMin", f"{rMin}",
                "--rMax", f"{rMax}",
                "--X-rtd", "MINIMIZER_freezeDisassociatedParams",
                "--X-rtd", "MINIMIZER_multiMin_hideConstants",
                "--X-rtd", "MINIMIZER_multiMin_maskConstraints",
                "--X-rtd", "MINIMIZER_multiMin_maskChannels=2",
                "-P", "r",
                "--floatOtherPOIs", "1",
                "--saveWorkspace",
                "--saveFitResult",
                "--snapshotName", "MultiDimFit",
                "--alignEdges", "1",
                "--firstPoint", f"{current_point}",
                "--lastPoint", f"{current_point}",
                "--setParameterRanges", f"{config['combine_fit']['setParameterRange']}",
                "--cminApproxPreFitTolerance", f"{cminApproxPreFitTolerance}",
                "-w", "w",
            ]
        else:
            saveSpecifiedIndex = ",".join(combineVariableDict(self.variable, self.year)['pdfIndeces'])
            paramStr = ",".join(combineVariableDict(self.variable, self.year)['paramStr'])
            arguments = [
                "combineTool.py",
                "-M", "MultiDimFit",
                "-d", firstStepPath,
                "--freezeParameters", "MH",
                "-m", f"{HIGGS_MASS}",
                "-n", f"DataPostFitScanFit_{current_cat}.POINTS.{current_point}.{current_point}",
                "--cminDefaultMinimizerStrategy=0",
                "--cminFallbackAlgo", "Minuit2,Migrad,1:10",
                "--algo", "grid",
                "--points", f"{n_points}",
                "--X-rtd", "MINIMIZER_freezeDisassociatedParams",
                "--X-rtd", "MINIMIZER_multiMin_hideConstants",
                "--X-rtd", "MINIMIZER_multiMin_maskConstraints",
                "--X-rtd", "MINIMIZER_multiMin_maskChannels=2",
                "-P", f"{current_cat}",
                "--saveWorkspace",
                "--saveFitResult",
                "--floatOtherPOIs", "1",
                "--snapshotName", "MultiDimFit",
                "--alignEdges", "1",
                "--firstPoint", f"{current_point}",
                "--lastPoint", f"{current_point}",
                "--saveSpecifiedIndex", saveSpecifiedIndex,
                "--setParameters", paramStr,
                "-w", "w",
            ]
            arguments.extend(_scan_parameter_range_args(config, current_cat))
        command = arguments
        try:
            result = subprocess.run(command, check=True, text=True, capture_output=True)
            print("Script output:", result.stdout)
            print("Script executed successfully.")
        except subprocess.CalledProcessError as e:
            print("Error executing script:", e.stderr)
        
        # Copy the files back to pnfs if we are on slurm/psi
        if self.batch_flavor == "slurm/psi":
            # Have to copy over the output to the final directory
            # Don't forget to VOMS!
            if "/work" in output_dir:
                slurm_copy_command = [
                    'cp', '-rf',
                    f"{os.environ['TARGET_PATH']}/Combine/",
                    output_dir
                ]
            else:
                slurm_copy_command = [
                    'xrdcp', '-rf',
                    f"{os.environ['TARGET_PATH']}/Combine/",
                    'root://t3dcachedb03.psi.ch:1094//'+output_dir
                ]
            print(slurm_copy_command)
            execute_command(slurm_copy_command)
            # Clean up the temporary directory
            shutil.rmtree(os.environ["TARGET_PATH"])
            
        os.chdir(cwd)
        
class UnblindedFitCategoryStat(Task, law.LocalWorkflow, HTCondorWorkflow, SlurmWorkflow): #(law.Task): #(Task, HTCondorWorkflow, law.LocalWorkflow):
    output_dir = law.Parameter(default = '', description="Path to the output directory")
    variable = law.Parameter(default="", description="Variable to be used")
    year = law.Parameter(default='2022', description="Year")
    nPoints = law.Parameter(default=30, description="Number of points for the LL scan")
    
    batch_flavor = law.Parameter(default="htcondor", description="Batch system to use")

    # def requires(self):
    def workflow_requires(self):
        workflow_reqs = super().workflow_requires()

        tasks = {}

        if workflow_reqs:
            tasks.update(workflow_reqs)
        
        config = self.get_input_config()
        output_dir = self.get_output_dir()
        
        fitConfig = config["combine_fit"]
  
        tasks["UnblindedFitStatSingle"] = UnblindedFitStatSingle(output_dir=output_dir, variable=self.variable, year=self.year, batch_flavor=self.batch_flavor, version=self.variable if self.variable != "" else "inclusive", workflow=fitConfig["execution"], slurm_partition=fitConfig['batchPartition'], slurm_memory=fitConfig['batchMemory'], slurm_max_runtime=fitConfig['batchMaxRuntime'], htcondor_partition=fitConfig['batchPartition'], htcondor_memory=fitConfig['batchMemory'], htcondor_max_runtime=fitConfig['batchMaxRuntime'])
        
        return tasks
    
    def create_branch_map(self):
        if self.variable == '':
            cats = ["r"]
        else:
            cats = combineVariableDict(self.variable, self.year)['paramStrNoOne']
        branch_data = [(cat, point) for cat in cats for point in range(int(self.nPoints))]
        branch_map = {i: current_branch for i, current_branch in enumerate(branch_data)}
        return branch_map

    def output(self):
        cat, current_point = self.branch_data
        
        config = self.get_input_config()
        output_dir = self.get_output_dir()

        if self.variable == '':
            fitFolderName = f'runFits_mu_fiducial'
        else:
            fitFolderName = f'runFits_{self.variable}'
        output = [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'dataFit', f'higgsCombineDataPostFitScanStat_{cat}.POINTS.{current_point}.{current_point}.MultiDimFit.mH{HIGGS_MASS}.root')]
        
        outputFileTargets = []
                
        for _, current_output_path in enumerate(output):
            outputFileTargets.append(law.LocalFileTarget(current_output_path))

        return outputFileTargets

    def run(self):
        cat, current_point = self.branch_data

        if self.variable == '':
            fitFolderName = f'runFits_mu_fiducial'
            
        else:
            fitFolderName = f'runFits_{self.variable}'
                    
        #Load central config file
        with open(configYamlPath, 'r') as file:
            config = yaml.safe_load(file)

        # default values, if not set in config
        cminApproxPreFitTolerance = config.get("combine_fit", {}).get("cminApproxPreFitTolerance", 0.01)
        rMin = config.get("combine_fit", {}).get("rMin", 0.7)
        rMax = config.get("combine_fit", {}).get("rMax", 1.6)

        if self.output_dir == '':
            output_dir = config['outputFolder']
        else:
            output_dir = self.output_dir  

        cwd = os.getcwd()
        if self.batch_flavor == "slurm/psi":
            # Have to use /scratch/batch_username/ for slurm/psi
            if "/work" in output_dir:
                execute_command([f'mkdir -p {output_dir}/Combine/{fitFolderName}/dataFit'], shell=True)
            else:   
                execute_command([f'xrdfs root://t3dcachedb03.psi.ch:1094/ mkdir -p {output_dir}/Combine/{fitFolderName}/dataFit'], shell=True)

            os.environ["TARGET_PATH"] = f"/scratch/{os.environ['USER']}/{os.environ['SLURM_JOB_ID']}"
            execute_command([f'mkdir -p $TARGET_PATH/Combine/{fitFolderName}/dataFit'], shell=True)
            os.chdir(os.path.join(os.environ["TARGET_PATH"], 'Combine', fitFolderName, 'dataFit'))
        else:
            execute_command([f'mkdir -p {output_dir}/Combine/{fitFolderName}/dataFit'], shell=True)
            os.chdir(os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'dataFit'))
    
        firstStepPath = os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'dataFit', f"higgsCombineDataPostFitBestFitStat_{cat}.MultiDimFit.mH{HIGGS_MASS}.root")
        local_first_step = os.path.join(os.getcwd(), os.path.basename(firstStepPath))
        if not os.path.exists(firstStepPath):
            raise RuntimeError(f"Required stat-only best fit snapshot not found: {firstStepPath}")
        if os.path.abspath(local_first_step) != os.path.abspath(firstStepPath):
            shutil.copy2(firstStepPath, local_first_step)
        input_path = local_first_step if os.path.exists(local_first_step) else firstStepPath
        n_points = int(self.nPoints)
                
        if self.variable == '':
            arguments = [
                "combine",
                "-M", "MultiDimFit",
                "-d", input_path,
                "--freezeParameters", "allConstrainedNuisances,MH",
                "-m", f"{HIGGS_MASS}",
                "-n", f"DataPostFitScanStat_{cat}.POINTS.{current_point}.{current_point}",
                "--cminDefaultMinimizerStrategy=0",
                "--cminFallbackAlgo", "Minuit2,Migrad,1:10",
                "--algo", "grid",
                "--rMin", f"{rMin}",
                "--rMax", f"{rMax}",
                "--points", f"{n_points}",
                "--X-rtd", "MINIMIZER_freezeDisassociatedParams",
                "--X-rtd", "MINIMIZER_multiMin_hideConstants",
                "--X-rtd", "MINIMIZER_multiMin_maskConstraints",
                "--X-rtd", "MINIMIZER_multiMin_maskChannels=2",
                "-P", f"{cat}",
                "--floatOtherPOIs", "1",
                "--saveWorkspace",
                "--saveFitResult",
                "--snapshotName", "MultiDimFit",
                "--alignEdges", "1",
                "--firstPoint", f"{current_point}",
                "--lastPoint", f"{current_point}",
                "--cminApproxPreFitTolerance", f"{cminApproxPreFitTolerance}",
                "-w", "w",
            ]
        else:
            arguments = [
                "combineTool.py",
                "-M", "MultiDimFit",
                "-d", input_path,
                "--freezeParameters", "allConstrainedNuisances,MH",
                "-m", f"{HIGGS_MASS}",
                "-n", f"DataPostFitScanStat_{cat}.POINTS.{current_point}.{current_point}",
                "--cminDefaultMinimizerStrategy=0",
                "--cminFallbackAlgo", "Minuit2,Migrad,1:10",
                "--algo", "grid",
                "--points", f"{n_points}",
                "--X-rtd", "MINIMIZER_freezeDisassociatedParams",
                "--X-rtd", "MINIMIZER_multiMin_hideConstants",
                "--X-rtd", "MINIMIZER_multiMin_maskConstraints",
                "--X-rtd", "MINIMIZER_multiMin_maskChannels=2",
                "-P", f"{cat}",
                "--saveWorkspace",
                "--saveFitResult",
                "--floatOtherPOIs", "1",
                "--snapshotName", "MultiDimFit",
                "--alignEdges", "1",
                "--firstPoint", f"{current_point}",
                "--lastPoint", f"{current_point}",
                "-w", "w",
            ]
            arguments.extend(_scan_parameter_range_args(config, cat))
        command = arguments
        try:
            result = subprocess.run(command, check=True, text=True, capture_output=True)
            print("Script output:", result.stdout)
            print("Script executed successfully.")
        except subprocess.CalledProcessError as e:
            print("Error executing script:", e.stderr)
            
        # Copy the files back to pnfs if we are on slurm/psi
        if self.batch_flavor == "slurm/psi":
            # Have to copy over the output to the final directory
            # Don't forget to VOMS!
            if "/work" in output_dir:
                slurm_copy_command = [
                    'cp', '-rf',
                    f"{os.environ['TARGET_PATH']}/Combine/",
                    output_dir
                ]
            else:
                slurm_copy_command = [
                    'xrdcp', '-rf',
                    f"{os.environ['TARGET_PATH']}/Combine/",
                    'root://t3dcachedb03.psi.ch:1094//'+output_dir
                ]
            print(slurm_copy_command)
            execute_command(slurm_copy_command)
            # Clean up the temporary directory
            shutil.rmtree(os.environ["TARGET_PATH"])
        
        os.chdir(cwd)
        
class CreateUnblindedFit(Task, law.LocalWorkflow, HTCondorWorkflow, SlurmWorkflow): #(law.Task): #(Task, HTCondorWorkflow, law.LocalWorkflow):
    output_dir = law.Parameter(default = '', description="Path to the output directory")
    variable = law.Parameter(default="", description="Variable to be used")
    year = law.Parameter(default='2022', description="Year")

    batch_flavor = law.Parameter(default="htcondor", description="Batch system to use")

    # def requires(self):
    def workflow_requires(self):
        workflow_reqs = super().workflow_requires()

        tasks = {}

        if workflow_reqs:
            tasks.update(workflow_reqs)
        
        config = self.get_input_config()
        output_dir = self.get_output_dir()

        fitConfig = config["combine_fit"]
            
        tasks["UnblindedFitCategorySyst"] = UnblindedFitCategorySyst(output_dir=output_dir, variable=self.variable, year=self.year, batch_flavor=self.batch_flavor, nPoints=fitConfig["unblindedFit_numPoints"], version=self.variable if self.variable != "" else "inclusive", workflow=fitConfig["execution"], slurm_partition=fitConfig['batchPartition'], slurm_memory=fitConfig['batchMemory'], slurm_max_runtime=fitConfig['batchMaxRuntime'], htcondor_partition=fitConfig['batchPartition'], htcondor_memory=fitConfig['batchMemory'], htcondor_max_runtime=fitConfig['batchMaxRuntime'])
        tasks["UnblindedFitCategoryStat"] = UnblindedFitCategoryStat(output_dir=output_dir, variable=self.variable, year=self.year, batch_flavor=self.batch_flavor, nPoints=fitConfig["unblindedFit_numPoints"], version=self.variable if self.variable != "" else "inclusive", workflow=fitConfig["execution"], slurm_partition=fitConfig['batchPartition'], slurm_memory=fitConfig['batchMemory'], slurm_max_runtime=fitConfig['batchMaxRuntime'], htcondor_partition=fitConfig['batchPartition'], htcondor_memory=fitConfig['batchMemory'], htcondor_max_runtime=fitConfig['batchMaxRuntime'])
        
        return tasks
    
    def create_branch_map(self):
        # map branch indexes to ascii numbers from 97 to 122 ("a" to "z")        
        branch_list = [0]
        
        branch_map = {i: branch for i, branch in enumerate(branch_list)}
        return branch_map

    def output(self):
        config = self.get_input_config()
        output_dir = self.get_output_dir()

        if self.variable == '':
            fitFolderName = f'runFits_mu_fiducial'
        else:
            fitFolderName = f'runFits_{self.variable}'
            
        output = []

        output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'dataFit', 'scans')]
        
        if self.variable == '':
            cat = "r"
            output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'dataFit', 'scans', f'scan_{cat}_observed.root')]
            output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'dataFit', 'scans', f'scan_{cat}_observed.pdf')]
            output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'dataFit', 'scans', f'scan_{cat}_observed.png')]
        else:
            for cat in combineVariableDict(self.variable, self.year)['paramStrNoOne']:
                
                output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'dataFit', 'scans', f'scan_{cat}_observed.root')]
                output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'dataFit', 'scans', f'scan_{cat}_observed.pdf')]
                output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'dataFit', 'scans', f'scan_{cat}_observed.png')]

        outputFileTargets = []
                
        for _, current_output_path in enumerate(output):
            outputFileTargets.append(law.LocalFileTarget(current_output_path))
            
        return outputFileTargets

    def run(self):
        
        if self.variable == '':
            fitFolderName = f'runFits_mu_fiducial'
        else:
            fitFolderName = f'runFits_{self.variable}'
            
                    
        config = self.get_input_config()
        output_dir = self.get_output_dir() 

        cwd = os.getcwd()
        if self.batch_flavor == "slurm/psi":
            # Have to use /scratch/batch_username/ for slurm/psi
            if "/work" in output_dir:
                execute_command([f'mkdir -p {output_dir}/Combine/{fitFolderName}/dataFit/scans'], shell=True)
            else:   
                execute_command([f'xrdfs root://t3dcachedb03.psi.ch:1094/ mkdir -p {output_dir}/Combine/{fitFolderName}/dataFit/scans'], shell=True)

            os.environ["TARGET_PATH"] = f"/scratch/{os.environ['USER']}/{os.environ['SLURM_JOB_ID']}"
            execute_command([f'mkdir -p $TARGET_PATH/Combine/{fitFolderName}/dataFit/scans'], shell=True)
            os.chdir(os.path.join(os.environ["TARGET_PATH"], 'Combine', fitFolderName, 'dataFit'))
        else:
            execute_command([f'mkdir -p {output_dir}/Combine/{fitFolderName}/dataFit/scans'], shell=True)
            os.chdir(os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'dataFit'))
        
        if self.variable == '':
            cats = ["r"]
        else:
            cats = combineVariableDict(self.variable, self.year)['paramStrNoOne']
        
        job_datafit_dir = os.path.join(os.environ["TARGET_PATH"], 'Combine', fitFolderName, 'dataFit') if self.batch_flavor == "slurm/psi" else os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'dataFit')

        if self.batch_flavor == "slurm/psi":
            # Have to copy over the input to the JOB directory
            # Don't forget to VOMS!
            if "/work" in output_dir:
                slurm_copy_command = [
                    'cp', '-rf',
                    f'{output_dir}/Combine/{fitFolderName}/dataFit',
                    f"{os.environ['TARGET_PATH']}/Combine/{fitFolderName}"
                ]
            else:
                slurm_copy_command = [
                    'xrdcp', '-rf',
                    'root://t3dcachedb03.psi.ch:1094//'+f'{output_dir}/Combine/{fitFolderName}/dataFit',
                    f"{os.environ['TARGET_PATH']}/Combine/{fitFolderName}"
                ]
            print(slurm_copy_command)
            execute_command(slurm_copy_command)

        n_points = int(config["combine_fit"]["unblindedFit_numPoints"])

        def hadd_scan_outputs(kind, current_cat):
            target = os.path.join(job_datafit_dir, f'higgsCombineDataPostFitScan{kind}_{current_cat}.MultiDimFit.mH{HIGGS_MASS}.root')
            sources = [
                os.path.join(job_datafit_dir, f'higgsCombineDataPostFitScan{kind}_{current_cat}.POINTS.{idx}.{idx}.MultiDimFit.mH{HIGGS_MASS}.root')
                for idx in range(n_points)
            ]
            command = ["hadd", "-f", target] + sources
            try:
                result = subprocess.run(command, check=True, text=True, capture_output=True)
                print("Script output:", result.stdout)
                print("Script executed successfully.")
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Error executing hadd for {kind} scan {current_cat}: {e.stderr}") from e

        for cat in cats:
            hadd_scan_outputs("Fit", cat)
            hadd_scan_outputs("Stat", cat)

        def ensure_scan_file(path):
            if (not os.path.exists(path)) or os.path.getsize(path) == 0:
                raise RuntimeError(f"Required scan output is missing: {path}")

        for cat in cats:
            fit_scan = os.path.join(job_datafit_dir, f'higgsCombineDataPostFitScanFit_{cat}.MultiDimFit.mH{HIGGS_MASS}.root')
            stat_scan = os.path.join(job_datafit_dir, f'higgsCombineDataPostFitScanStat_{cat}.MultiDimFit.mH{HIGGS_MASS}.root')
            ensure_scan_file(fit_scan)
            ensure_scan_file(stat_scan)

        for cat in cats:
            arguments = [
                "plot1DScan.py",
                os.path.join(job_datafit_dir, f'higgsCombineDataPostFitScanFit_{cat}.MultiDimFit.mH{HIGGS_MASS}.root'),
                "-o", f"scans/scan_{cat}_observed",
                "--POI", f"{cat}",
                "--others", os.path.join(job_datafit_dir, f'higgsCombineDataPostFitScanStat_{cat}.MultiDimFit.mH{HIGGS_MASS}.root')+":stat-only:2",
                "--main-label", "Observed",
                "--translate", os.path.join(os.environ["ANALYSIS_PATH"], 'Combine', 'pois.json')
            ]
            command = arguments
            # print(command)
            try:
                result = subprocess.run(command, check=True, text=True, capture_output=True)
                print("Script output:", result.stdout)
                print("Script executed successfully.")
            except subprocess.CalledProcessError as e:
                print("Error executing script:", e.stderr)
        
        # Copy the files back to pnfs if we are on slurm/psi
        if self.batch_flavor == "slurm/psi":
            # Have to copy over the output to the final directory
            # Don't forget to VOMS!
            if "/work" in output_dir:
                slurm_copy_command = [
                    'cp', '-rf',
                    f"{os.environ['TARGET_PATH']}/Combine/",
                    output_dir
                ]
            else:
                slurm_copy_command = [
                    'xrdcp', '-rf',
                    f"{os.environ['TARGET_PATH']}/Combine/",
                    'root://t3dcachedb03.psi.ch:1094//'+output_dir
                ]
            print(slurm_copy_command)
            execute_command(slurm_copy_command)
            # Clean up the temporary directory
            shutil.rmtree(os.environ["TARGET_PATH"])
                
        os.chdir(cwd)

class UnblindedCovCorrHesse(Task, law.LocalWorkflow, HTCondorWorkflow, SlurmWorkflow): #(law.Task): #(Task, HTCondorWorkflow, law.LocalWorkflow):
    output_dir = law.Parameter(default = '', description="Path to the output directory")
    variable = law.Parameter(default="", description="Variable to be used")
    year = law.Parameter(default='2022', description="Year")

    batch_flavor = law.Parameter(default="htcondor", description="Batch system to use")

    # def requires(self):
    def workflow_requires(self):
        workflow_reqs = super().workflow_requires()

        tasks = {}

        if workflow_reqs:
            tasks.update(workflow_reqs)
        
        config = self.get_input_config()
        output_dir = self.get_output_dir()

        tasks["RunT2WS"] = RunText2Workspace(output_dir=output_dir, variable=self.variable, year=self.year, batch_flavor=self.batch_flavor, workflow=config["combine_fit"]["execution"], version=self.variable if self.variable != "" else "inclusive", slurm_partition=config["combine_fit"]['batchPartition'], slurm_memory=config["combine_fit"]['batchMemory'], slurm_max_runtime=config["combine_fit"]['batchMaxRuntime'], htcondor_partition=config["combine_fit"]['batchPartition'], htcondor_memory=config["combine_fit"]['batchMemory'], htcondor_max_runtime=config["combine_fit"]['batchMaxRuntime'])
        
        return tasks

    def create_branch_map(self):
        branch_map = {i: current_branch for i, current_branch in enumerate([0])}
        return branch_map

    def output(self):        
        config = self.get_input_config()
        output_dir = self.get_output_dir()

        if self.variable == '':
            fitFolderName = f'runFits_mu_fiducial'
        else:
            fitFolderName = f'runFits_{self.variable}'
                  
        if self.variable == '':
            output = []
        else:
            output = [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'hesse')]

            output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'hesse', f'robustHessefirstStep_data.root')]
            output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'hesse', f'multidimfitfirstStep_data.root')]
            output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'hesse', f'higgsCombinefirstStep_data.MultiDimFit.mH{HIGGS_MASS}.root')]
            
        outputFileTargets = []
                
        for _, current_output_path in enumerate(output):
            outputFileTargets.append(law.LocalFileTarget(current_output_path))

        return outputFileTargets

    def run(self):
       
        if self.variable == '':
            print("Running UnblindedCovCorrHesse for inclusive does not make sense. Please specify a variable.")
            exit(1)
            
        else:
            fitFolderName = f'runFits_{self.variable}'
                  
        config = self.get_input_config()
        output_dir = self.get_output_dir() 
            
        if self.variable == '':
            datacard_path = os.path.join(output_dir, 'Combine', outdir, f'Datacard_{self.year}.root')
        else:
            datacard_path = os.path.join(output_dir, 'Combine', outdir, f'Datacard_{self.variable}_{self.year}.root')

        cwd = os.getcwd()
        
        if self.batch_flavor == "slurm/psi":
            # Have to use /scratch/batch_username/ for slurm/psi
            if "/work" in output_dir:
                execute_command([f'mkdir -p {output_dir}/Combine/{fitFolderName}/hesse'], shell=True)
            else:
                execute_command([f'xrdfs root://t3dcachedb03.psi.ch:1094/ mkdir -p {output_dir}/Combine/{fitFolderName}/hesse'], shell=True)

            os.environ["TARGET_PATH"] = f"/scratch/{os.environ['USER']}/{os.environ['SLURM_JOB_ID']}"
            execute_command([f'mkdir -p $TARGET_PATH/Combine/{fitFolderName}/hesse'], shell=True)
            os.chdir(os.path.join(os.environ["TARGET_PATH"], 'Combine', fitFolderName, 'hesse'))
        else:
            execute_command([f'mkdir -p {output_dir}/Combine/{fitFolderName}/hesse'], shell=True)
            os.chdir(os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'hesse'))

        arguments = [
            "combine",
            "-M", "MultiDimFit",
            datacard_path,
            "--freezeParameters", "MH",
            "-m", f"{HIGGS_MASS}",
            "-n", "firstStep_data",
            "--saveWorkspace",
            "--saveFitResult",
            "--saveSpecifiedIndex", f"""{",".join(combineVariableDict(self.variable, self.year)['pdfIndeces'])}""",
            "--floatOtherPOIs", "1",
            "--robustHesse", "1",
            "--robustHesseSave", "1",
            "--cminDefaultMinimizerStrategy=0",
            "--cminFallbackAlgo", "Minuit2,Migrad,1:10",
            "--X-rtd", "MINIMIZER_freezeDisassociatedParams",
            "--X-rtd", "MINIMIZER_multiMin_hideConstants",
            "--X-rtd", "MINIMIZER_multiMin_maskConstraints",
            "--X-rtd", "MINIMIZER_multiMin_maskChannels=2"
        ]
        command = arguments
        print(command)
        try:
            result = subprocess.run(command, check=True, text=True, capture_output=True)
            print("Script output:", result.stdout)
            print("Script executed successfully.")
        except subprocess.CalledProcessError as e:
            print("Error executing script:", e.stderr)
        
        # Copy the files back to pnfs if we are on slurm/psi
        if self.batch_flavor == "slurm/psi":
            # Have to copy over the output to the final directory
            # Don't forget to VOMS!
            if "/work" in output_dir:
                slurm_copy_command = [
                    'cp', '-rf',
                    f"{os.environ['TARGET_PATH']}/Combine/",
                    output_dir
                ]
            else:
                slurm_copy_command = [
                    'xrdcp', '-rf',
                    f"{os.environ['TARGET_PATH']}/Combine/",
                    'root://t3dcachedb03.psi.ch:1094//'+output_dir
                ]
            print(slurm_copy_command)
            execute_command(slurm_copy_command)
            # Clean up the temporary directory
            shutil.rmtree(os.environ["TARGET_PATH"])
            
        os.chdir(cwd)
        
class UnblindedCovCorr(Task, law.LocalWorkflow, HTCondorWorkflow, SlurmWorkflow): #(law.Task): #(Task, HTCondorWorkflow, law.LocalWorkflow):
    output_dir = law.Parameter(default = '', description="Path to the output directory")
    variable = law.Parameter(default="", description="Variable to be used")
    year = law.Parameter(default='2022', description="Year")
    noPreliminary = law.Parameter(default=False, description="Flag, if final plot should bear the Preliminary.")

    batch_flavor = law.Parameter(default="htcondor", description="Batch system to use")

    # def requires(self):
    def workflow_requires(self):
        workflow_reqs = super().workflow_requires()

        tasks = {}

        if workflow_reqs:
            tasks.update(workflow_reqs)
        
        config = self.get_input_config()
        output_dir = self.get_output_dir()
            
        if self.variable == '':
            version = 'r'
        else:
            version = self.variable

        corrConfig = config["combine_hesse"]
            
        tasks["UnblindedCovCorrHesse"] = UnblindedCovCorrHesse(output_dir=output_dir, variable=self.variable, year=self.year, version=version, workflow=corrConfig["execution"], slurm_partition=corrConfig['batchPartition'], slurm_memory=corrConfig['batchMemory'], slurm_max_runtime=corrConfig['batchMaxRuntime'], htcondor_partition=corrConfig['batchPartition'], htcondor_memory=corrConfig['batchMemory'], htcondor_max_runtime=corrConfig['batchMaxRuntime'], batch_flavor=self.batch_flavor)
        
        return tasks

    def create_branch_map(self):
        branch_map = {i: current_branch for i, current_branch in enumerate([0])}
        return branch_map

    def output(self):        
        config = self.get_input_config()
        output_dir = self.get_output_dir()

        if self.variable == '':
            fitFolderName = f'runFits_mu_fiducial'
        else:
            fitFolderName = f'runFits_{self.variable}'
            
        # output = [os.path.join(output_dir, 'Combine', outdir, fitFolderName)]
        output = [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'hesse', 'Plots', 'data')]

        output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'hesse', 'Plots', 'data', f'corrMatrix_{self.variable}_syst_obs.pdf')]
        output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'hesse', 'Plots', 'data', f'corrMatrix_{self.variable}_syst_obs.png')]
        output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'hesse', 'Plots', 'data', f'covMatrix_{self.variable}_syst_obs.pdf')]
        output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'hesse', 'Plots', 'data', f'covMatrix_{self.variable}_syst_obs.png')]
        
        outputFileTargets = []
                
        for _, current_output_path in enumerate(output):
            outputFileTargets.append(law.LocalFileTarget(current_output_path))
            
        # print(outputFileTargets)

        return outputFileTargets

    def run(self):
       
        if self.variable == '':
            print("Running UnblindedCovCorrHesse for inclusive does not make sense. Please specify a variable.")
            exit(1)
            
        else:
            fitFolderName = f'runFits_{self.variable}'
                  
        config = self.get_input_config()
        output_dir = self.get_output_dir() 

        if self.batch_flavor == "slurm/psi":
            # Have to use /scratch/batch_username/ for slurm/psi
            if "/work" in output_dir:
                execute_command([f'mkdir -p {output_dir}/Combine/{fitFolderName}/hesse/Plots/data'], shell=True)
            else:   
                execute_command([f'xrdfs root://t3dcachedb03.psi.ch:1094/ mkdir -p {output_dir}/Combine/{fitFolderName}/hesse/Plots/data'], shell=True)

            os.environ["TARGET_PATH"] = f"/scratch/{os.environ['USER']}/{os.environ['SLURM_JOB_ID']}"
            execute_command([f'mkdir -p $TARGET_PATH/Combine/{fitFolderName}/hesse/Plots/data'], shell=True)

            if "/work" in output_dir:
                slurm_copy_command = [
                    'cp', '-rf',
                    f'{output_dir}/Combine/{fitFolderName}/hesse',
                    f'{os.environ["TARGET_PATH"]}/Combine/{fitFolderName}/'
                ]
            else:
                slurm_copy_command = [
                    'xrdcp', '-rf',
                    'root://t3dcachedb03.psi.ch:1094//'+ f'{output_dir}/Combine/{fitFolderName}/hesse',
                    f'{os.environ["TARGET_PATH"]}/Combine/{fitFolderName}/'
                ]
            execute_command(slurm_copy_command)
            # output_dir = os.path.join(os.environ["TARGET_PATH"], 'Combine', fitFolderName, 'hesse', 'Plots', 'data')
            temp_output_dir = os.environ["TARGET_PATH"]
        else:
            execute_command([f'mkdir -p {output_dir}/Combine/{fitFolderName}/hesse/Plots/data'], shell=True)
            temp_output_dir = output_dir
            # output_dir = os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'hesse', 'Plots', 'data')     
        
        cwd = os.getcwd()
        os.chdir(os.path.join(os.environ["ANALYSIS_PATH"], 'Plots'))

        arguments = [
            "python3",
            f"{os.path.join(os.environ['ANALYSIS_PATH'], 'Plots', 'makeCorrMatrix.py')}",
            "--inputJson", f"{os.path.join(os.environ['ANALYSIS_PATH'], 'Plots', 'inputs_robustHesse.json')}",
            "--mode", f"{self.variable}",
            "--input", f"{os.path.join(temp_output_dir, 'Combine', outdir, fitFolderName, 'hesse', f'robustHessefirstStep_data.root')}",
            "--output", f"{os.path.join(temp_output_dir, 'Combine', outdir, fitFolderName, 'hesse', 'Plots', 'data')}",
            "--translate", f"{os.path.join(os.environ['ANALYSIS_PATH'], 'Plots', 'poi_differential_hesse_noLabels.json')}",
            "--doObserved"
        ]
        if convert_boolean_string(self.noPreliminary):
            arguments.append("--noPreliminary")
        command = arguments
        # print(command)
        try:
            result = subprocess.run(command, check=True, text=True, capture_output=True)
            print("Script output:", result.stdout)
            print("Script executed successfully.")
        except subprocess.CalledProcessError as e:
            print("Error executing script:", e.stderr)
            
        arguments = [
            "python3",
            f"{os.path.join(os.environ['ANALYSIS_PATH'], 'Plots', 'makeCorrMatrix.py')}",
            "--inputJson", f"{os.path.join(os.environ['ANALYSIS_PATH'], 'Plots', 'inputs_robustHesse.json')}",
            "--mode", f"{self.variable}",
            "--input", f"{os.path.join(temp_output_dir, 'Combine', outdir, fitFolderName, 'hesse', f'robustHessefirstStep_data.root')}",
            "--output", f"{os.path.join(temp_output_dir, 'Combine', outdir, fitFolderName, 'hesse', 'Plots', 'data')}",
            "--translate", f"{os.path.join(os.environ['ANALYSIS_PATH'], 'Plots', 'poi_differential_hesse_noLabels.json')}",
            "--doCov",
            "--doObserved"
        ]
        if convert_boolean_string(self.noPreliminary):
            arguments.append("--noPreliminary")
        command = arguments
        # print(command)
        try:
            result = subprocess.run(command, check=True, text=True, capture_output=True)
            print("Script output:", result.stdout)
            print("Script executed successfully.")
        except subprocess.CalledProcessError as e:
            print("Error executing script:", e.stderr)
            
        # Copy the files back to pnfs if we are on slurm/psi
        if self.batch_flavor == "slurm/psi":
            # Have to copy over the output to the final directory
            # Don't forget to VOMS!
            if "/work" in output_dir:
                slurm_copy_command = [
                    'cp', '-rf',
                    f"{os.environ['TARGET_PATH']}/Combine/",
                    output_dir
                ]
            else:
                slurm_copy_command = [
                    'xrdcp', '-rf',
                    f"{os.environ['TARGET_PATH']}/Combine/",
                    'root://t3dcachedb03.psi.ch:1094//'+output_dir
                ]
            print(slurm_copy_command)
            execute_command(slurm_copy_command)
            # Clean up the temporary directory
            shutil.rmtree(os.environ["TARGET_PATH"])    
        
        os.chdir(cwd)
        
        
class UnblindedImpactFirstStep(Task, law.LocalWorkflow, HTCondorWorkflow, SlurmWorkflow): #(law.Task): #(Task, HTCondorWorkflow, law.LocalWorkflow):
    output_dir = law.Parameter(default = '', description="Path to the output directory")
    variable = law.Parameter(default="", description="Variable to be used")
    year = law.Parameter(default='2022', description="Year")

    batch_flavor = law.Parameter(default="htcondor", description="Batch system to use")

    # def requires(self):
    def workflow_requires(self):
        workflow_reqs = super().workflow_requires()

        tasks = {}

        if workflow_reqs:
            tasks.update(workflow_reqs)
        
        config = self.get_input_config()
        output_dir = self.get_output_dir()

        impactConfig = config["combine_impacts"]

        tasks["RunT2WS"] = RunText2Workspace(output_dir=output_dir, variable=self.variable, year=self.year, version=self.variable if self.variable != "" else "inclusive", workflow=impactConfig["execution"], batch_flavor=self.batch_flavor, slurm_partition=impactConfig['batchPartition'], slurm_memory=impactConfig['batchMemory'], slurm_max_runtime=impactConfig['batchMaxRuntime'], htcondor_partition=impactConfig['batchPartition'], htcondor_memory=impactConfig['batchMemory'], htcondor_max_runtime=impactConfig['batchMaxRuntime'])

        return tasks

    def create_branch_map(self):
        branch_map = {i: current_branch for i, current_branch in enumerate([0])}
        return branch_map

    def output(self):        
        config = self.get_input_config()
        output_dir = self.get_output_dir()

        if self.variable == '':
            fitFolderName = f'runFits_mu_fiducial'
        else:
            fitFolderName = f'runFits_{self.variable}'
            
        output = [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'impact', 'unblinded')]

        output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'impact', 'unblinded', f'higgsCombine_initialFit_Test.MultiDimFit.mH{HIGGS_MASS}.root')]
        
        outputFileTargets = []
                
        for _, current_output_path in enumerate(output):
            outputFileTargets.append(law.LocalFileTarget(current_output_path))

        return outputFileTargets

    def run(self):
       
        if self.variable == '':
            fitFolderName = f'runFits_mu_fiducial'
            
        else:
            fitFolderName = f'runFits_{self.variable}'
                  
        #Load central config file
        with open(configYamlPath, 'r') as file:
            config = yaml.safe_load(file)

        # default values, if not set in config
        cminApproxPreFitTolerance = config.get("combine_impacts", {}).get("cminApproxPreFitTolerance", 0.01)
        setParameters = config.get("combine_impacts", {}).get("setParameters", 1.000)

        if self.output_dir == '':
            output_dir = config['outputFolder']
        else:
            output_dir = self.output_dir  
            
        if self.variable == '':
            datacard_path = os.path.join(output_dir, 'Combine', outdir, f'Datacard_{self.year}.root')
        else:
            datacard_path = os.path.join(output_dir, 'Combine', outdir, f'Datacard_{self.variable}_{self.year}.root')
        
        cwd = os.getcwd()

        if self.batch_flavor == "slurm/psi":
            # Have to use /scratch/batch_username/ for slurm/psi
            if "/work" in output_dir:
                execute_command([f'mkdir -p {output_dir}/Combine/{fitFolderName}/impact/unblinded'], shell=True)
            else:   
                execute_command([f'xrdfs root://t3dcachedb03.psi.ch:1094/ mkdir -p {output_dir}/Combine/{fitFolderName}/impact/unblinded'], shell=True)

            os.environ["TARGET_PATH"] = f"/scratch/{os.environ['USER']}/{os.environ['SLURM_JOB_ID']}"
            execute_command([f'mkdir -p $TARGET_PATH/Combine/{fitFolderName}/impact/unblinded'], shell=True)
            os.chdir(os.path.join(os.environ["TARGET_PATH"], 'Combine', fitFolderName, 'impact', 'unblinded'))
        else:
            execute_command([f'mkdir -p {output_dir}/Combine/{fitFolderName}/impact/unblinded'], shell=True)
            os.chdir(os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'impact', 'unblinded'))
                    
        if self.variable == '':
            arguments = [
                "combineTool.py",
                "-M", "Impacts",
                "-d", datacard_path,
                "--doInitialFit",
                "--robustFit", "1",
                "--freezeParameters", "MH",
                "-m", f"{HIGGS_MASS}",
                "--cminDefaultMinimizerStrategy=0",
                "--cminFallbackAlgo", "Minuit2,Migrad,1:10",
                "--X-rtd", "MINIMIZER_freezeDisassociatedParams",
                "--X-rtd", "MINIMIZER_multiMin_hideConstants",
                "--X-rtd", "MINIMIZER_multiMin_maskConstraints",
                "--X-rtd", "MINIMIZER_multiMin_maskChannels=2",
                "--cminApproxPreFitTolerance", f"{cminApproxPreFitTolerance}",
                "--setParameter", f"{setParameters}",
            ]
            command = arguments
            # print(command)
            try:
                result = subprocess.run(command, check=True, text=True, capture_output=True)
                print("Script output:", result.stdout)
                print("Script executed successfully.")
            except subprocess.CalledProcessError as e:
                print("Error executing script:", e.stderr)
        else:
            arguments = [
                "combine",
                "-M", "MultiDimFit",
                "-d", datacard_path,
                "--algo", "singles",
                "--redefineSignalPOIs", f"""{",".join(combineVariableDict(self.variable, self.year)['paramStrNoOne'])}""",
                "--freezeParameters", "MH",
                "-m", f"{HIGGS_MASS}",
                "--robustFit", "1",
                "-n", "_initialFit_Test",
                "--cminDefaultMinimizerStrategy=0",
                "--cminFallbackAlgo", "Minuit2,Migrad,1:10",
                "--X-rtd", "MINIMIZER_freezeDisassociatedParams",
                "--X-rtd", "MINIMIZER_multiMin_hideConstants",
                "--X-rtd", "MINIMIZER_multiMin_maskConstraints",
                "--X-rtd", "MINIMIZER_multiMin_maskChannels=2"
            ]
            command = arguments
            # print(command)
            try:
                result = subprocess.run(command, check=True, text=True, capture_output=True)
                print("Script output:", result.stdout)
                print("Script executed successfully.")
            except subprocess.CalledProcessError as e:
                print("Error executing script:", e.stderr)

        # Copy the files back to pnfs if we are on slurm/psi
        if self.batch_flavor == "slurm/psi":
            # Have to copy over the output to the final directory
            # Don't forget to VOMS!
            if "/work" in output_dir:
                slurm_copy_command = [
                    'cp', '-rf',
                    f"{os.environ['TARGET_PATH']}/Combine/",
                    output_dir
                ]
            else:
                slurm_copy_command = [
                    'xrdcp', '-rf',
                    f"{os.environ['TARGET_PATH']}/Combine/",
                    'root://t3dcachedb03.psi.ch:1094//'+output_dir
                ]
            print(slurm_copy_command)
            execute_command(slurm_copy_command)
            # Clean up the temporary directory
            shutil.rmtree(os.environ["TARGET_PATH"])   
            
        os.chdir(cwd)
        
class UnblindedImpactSecondStep(Task, law.LocalWorkflow, HTCondorWorkflow, SlurmWorkflow): #(law.Task): #(Task, HTCondorWorkflow, law.LocalWorkflow):
    output_dir = law.Parameter(default = '', description="Path to the output directory")
    variable = law.Parameter(default="", description="Variable to be used")
    year = law.Parameter(default='2022', description="Year")

    batch_flavor = law.Parameter(default="htcondor", description="Batch system to use")

    # def requires(self):
    def workflow_requires(self):
        workflow_reqs = super().workflow_requires()

        tasks = {}

        if workflow_reqs:
            tasks.update(workflow_reqs)
        
        config = self.get_input_config()
        output_dir = self.get_output_dir()

        impactConfig = config["combine_impacts"]
            
        tasks["UnblindedImpactFirstStep"] = UnblindedImpactFirstStep(output_dir=output_dir, variable=self.variable, year=self.year, batch_flavor=self.batch_flavor, workflow=impactConfig["execution"], version=self.variable if self.variable != "" else "inclusive", slurm_partition=impactConfig['batchPartition'], slurm_memory=impactConfig['batchMemory'], slurm_max_runtime=impactConfig['batchMaxRuntime'], htcondor_partition=impactConfig['batchPartition'], htcondor_memory=impactConfig['batchMemory'], htcondor_max_runtime=impactConfig['batchMaxRuntime'])
        
        return tasks

    def create_branch_map(self):
        

        config = self.get_input_config()
        output_dir = self.get_output_dir() 
            
        if self.variable == '':
            datacard_path = os.path.join(output_dir, 'Combine', outdir, f'Datacard_{self.year}.root')
        else:
            datacard_path = os.path.join(output_dir, 'Combine', outdir, f'Datacard_{self.variable}_{self.year}.root')
    
        # As seen in CMSSW_14_1_0_pre4/src/CombineHarvester/CombineTools/python/combine/Impacts.py
        def all_free_parameters(file, wsp, mc, pois):
            res = []
            wsFile = ROOT.TFile.Open(file)
            w = wsFile.Get(wsp)
            config = w.genobj(mc)
            pdfvars = config.GetPdf().getParameters(config.GetObservables())
            it = pdfvars.createIterator()
            var = it.Next()
            while var:
                if var.GetName() not in pois and (not var.isConstant()) and var.InheritsFrom("RooRealVar"):
                    res.append(var.GetName())
                var = it.Next()
            return res

        if self.variable == '':
            poiList = ["r"]
        else:
            poiList = combineVariableDict(self.variable, self.year)['paramStrNoOne']
        
        paramList = all_free_parameters(datacard_path, 'w', 'ModelConfig', poiList)

        
        branch_map = {i: current_param for i, current_param in enumerate(paramList)}
        return branch_map

    def output(self):        
        current_param = self.branch_data
        
        config = self.get_input_config()
        output_dir = self.get_output_dir()

        if self.variable == '':
            fitFolderName = f'runFits_mu_fiducial'
        else:
            fitFolderName = f'runFits_{self.variable}'
            
        # output = [os.path.join(output_dir, 'Combine', outdir, fitFolderName)]
        output = [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'impact', 'unblinded')]

        output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'impact', 'unblinded', f'higgsCombine_paramFit_Test_{current_param}.MultiDimFit.mH{HIGGS_MASS}.root')]
        
        outputFileTargets = []
                
        for _, current_output_path in enumerate(output):
            outputFileTargets.append(law.LocalFileTarget(current_output_path))
            
        # print(outputFileTargets)

        return outputFileTargets

    def run(self):
        current_param = self.branch_data
       
        if self.variable == '':
            fitFolderName = f'runFits_mu_fiducial'
            
        else:
            fitFolderName = f'runFits_{self.variable}'
                  
        #Load central config file
        with open(configYamlPath, 'r') as file:
            config = yaml.safe_load(file)

        # default values, if not set in config
        cminApproxPreFitTolerance = config.get("combine_impacts", {}).get("cminApproxPreFitTolerance", 0.01)
        stepSize = config.get("combine_impacts", {}).get("stepSize", 0.05)
        setCrossingTolerance = config.get("combine_impacts", {}).get("setCrossingTolerance", 0.00005)

        if self.output_dir == '':
            output_dir = config['outputFolder']
        else:
            output_dir = self.output_dir  
            
        if self.variable == '':
            datacard_path = os.path.join(output_dir, 'Combine', outdir, f'Datacard_{self.year}.root')
        else:
            datacard_path = os.path.join(output_dir, 'Combine', outdir, f'Datacard_{self.variable}_{self.year}.root')
        
        cwd = os.getcwd()

        if self.batch_flavor == "slurm/psi":
            # Have to use /scratch/batch_username/ for slurm/psi
            if "/work" in output_dir:
                execute_command([f'mkdir -p {output_dir}/Combine/{fitFolderName}/impact/unblinded'], shell=True)
            else:   
                execute_command([f'xrdfs root://t3dcachedb03.psi.ch:1094/ mkdir -p {output_dir}/Combine/{fitFolderName}/impact/unblinded'], shell=True)

            os.environ["TARGET_PATH"] = f"/scratch/{os.environ['USER']}/{os.environ['SLURM_JOB_ID']}"
            execute_command([f'mkdir -p $TARGET_PATH/Combine/{fitFolderName}/impact/unblinded'], shell=True)
            os.chdir(os.path.join(os.environ["TARGET_PATH"], 'Combine', fitFolderName, 'impact', 'unblinded'))
        else:
            execute_command([f'mkdir -p {output_dir}/Combine/{fitFolderName}/impact/unblinded'], shell=True)
            os.chdir(os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'impact', 'unblinded'))

        if self.variable == '':
            
            initial_fit = os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'impact', 'unblinded', f'higgsCombine_initialFit_Test.MultiDimFit.mH{HIGGS_MASS}.root')
            
            f = ROOT.TFile(initial_fit)
            tree = f.Get("limit")
            
            if not tree:
                print("Error: Tree 'limit' not found in the file.")
                exit(1)
            # Access the branch 'r_YH_2p0_2p5' and get its first value
            if hasattr(tree, 'r'):
                tree.GetEntry(0)  # Load the first entry
                poi_bf_value = getattr(tree, 'r')  # Access the branch value
                poi_bf_string = f'r={poi_bf_value}'
                print(f"First value of branch 'r': {poi_bf_value}")
            else:
                print("Error: Branch 'r' not found in the tree.")
                exit(1)

            arguments = [
                "combine",
                "-M", "MultiDimFit",
                "-d", datacard_path,
                "--algo", "impact",
                "--redefineSignalPOIs", "r",
                "--freezeParameters", "MH",
                "-m", f"{HIGGS_MASS}",
                "-P", f"{current_param}",
                "--setParameters", poi_bf_string,
                "--floatOtherPOIs", "1",
                "--saveInactivePOI", "1",
                "--robustFit", "1",
                "-n", f"_paramFit_Test_{current_param}",
                "--cminDefaultMinimizerStrategy=0",
                "--cminFallbackAlgo", "Minuit2,Migrad,1:10",
                "--X-rtd", "MINIMIZER_freezeDisassociatedParams",
                "--X-rtd", "MINIMIZER_multiMin_hideConstants",
                "--X-rtd", "MINIMIZER_multiMin_maskConstraints",
                "--X-rtd", "MINIMIZER_multiMin_maskChannels=2",
                "--cminApproxPreFitTolerance", f"{cminApproxPreFitTolerance}",
                "--stepSize", f"{stepSize}",
                "--setCrossingTolerance", f"{setCrossingTolerance}",
                "--robustHesse", "1"
            ]
            command = arguments
            # print(command)
            try:
                result = subprocess.run(command, check=True, text=True, capture_output=True)
                print("Script output:", result.stdout)
                print("Script executed successfully.")
            except subprocess.CalledProcessError as e:
                print("Error executing script:", e.stderr)
        else:

            initial_fit = os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'impact', 'unblinded', f'higgsCombine_initialFit_Test.MultiDimFit.mH{HIGGS_MASS}.root')

            poi_bf = []
            
            f = ROOT.TFile(initial_fit)
            tree = f.Get("limit")
            
            if not tree:
                print("Error: Tree 'limit' not found in the file.")
                exit(1)

            for poi in combineVariableDict(self.variable, self.year)['paramStrNoOne']:
                # Access the branch 'r_YH_2p0_2p5' and get its first value
                if hasattr(tree, poi):
                    tree.GetEntry(0)  # Load the first entry
                    first_value = getattr(tree, poi)  # Access the branch value
                    poi_bf.append(f'{poi}={first_value}')
                    print(f"First value of branch '{poi}': {first_value}")
                else:
                    print(f"Error: Branch '{poi}' not found in the tree.")
                    exit(1)

            poi_bf_string = ",".join(poi_bf)
        
            arguments = [
                "combine",
                "-M", "MultiDimFit",
                "-d", datacard_path,
                "--algo", "impact",
                "--redefineSignalPOIs", f"""{",".join(combineVariableDict(self.variable, self.year)['paramStrNoOne'])}""",
                "--setParameters", poi_bf_string,
                "--freezeParameters", "MH",
                "-m", f"{HIGGS_MASS}",
                "-P", f"{current_param}",
                "--floatOtherPOIs", "1",
                "--saveInactivePOI", "1",
                "--robustFit", "1",
                "-n", f"_paramFit_Test_{current_param}",
                "--cminDefaultMinimizerStrategy=0",
                "--cminFallbackAlgo", "Minuit2,Migrad,1:10",
                "--X-rtd", "MINIMIZER_freezeDisassociatedParams",
                "--X-rtd", "MINIMIZER_multiMin_hideConstants",
                "--X-rtd", "MINIMIZER_multiMin_maskConstraints",
                "--X-rtd", "MINIMIZER_multiMin_maskChannels=2"
            ]
            command = arguments
            # print(command)
            try:
                result = subprocess.run(command, check=True, text=True, capture_output=True)
                print("Script output:", result.stdout)
                print("Script executed successfully.")
            except subprocess.CalledProcessError as e:
                print("Error executing script:", e.stderr)

        # Copy the files back to pnfs if we are on slurm/psi
        if self.batch_flavor == "slurm/psi":
            # Have to copy over the output to the final directory
            # Don't forget to VOMS!
            if "/work" in output_dir:
                slurm_copy_command = [
                    'cp', '-rf',
                    f"{os.environ['TARGET_PATH']}/Combine/",
                    output_dir
                ]
            else:
                slurm_copy_command = [
                    'xrdcp', '-rf',
                    f"{os.environ['TARGET_PATH']}/Combine/",
                    'root://t3dcachedb03.psi.ch:1094//'+output_dir
                ]
            print(slurm_copy_command)
            execute_command(slurm_copy_command)
            # Clean up the temporary directory
            shutil.rmtree(os.environ["TARGET_PATH"])
            
        os.chdir(cwd)
        
class UnblindedImpactThirdStep(Task, law.LocalWorkflow, HTCondorWorkflow, SlurmWorkflow): #(law.Task): #(Task, HTCondorWorkflow, law.LocalWorkflow):
    output_dir = law.Parameter(default = '', description="Path to the output directory")
    variable = law.Parameter(default="", description="Variable to be used")
    year = law.Parameter(default='2022', description="Year")

    batch_flavor = law.Parameter(default="htcondor", description="Batch system to use")

    # def requires(self):
    def workflow_requires(self):
        workflow_reqs = super().workflow_requires()

        tasks = {}

        if workflow_reqs:
            tasks.update(workflow_reqs)
        
        if self.variable == '':
            version = 'r'
        else:
            version = self.variable
        
        config = self.get_input_config()
        output_dir = self.get_output_dir()

        impactConfig = config["combine_impacts"]
            
        tasks["UnblindedImpactSecondStep"] = UnblindedImpactSecondStep(output_dir=output_dir, variable=self.variable, year=self.year, version=version, workflow=impactConfig["execution"], slurm_partition=impactConfig['batchPartition'], slurm_memory=impactConfig['batchMemory'], slurm_max_runtime=impactConfig['batchMaxRuntime'], htcondor_partition=impactConfig['batchPartition'], htcondor_memory=impactConfig['batchMemory'], htcondor_max_runtime=impactConfig['batchMaxRuntime'], batch_flavor=self.batch_flavor)
        
        return tasks

    def create_branch_map(self):

        branch_map = {i: current_branch for i, current_branch in enumerate([0])}
        return branch_map

    def output(self):
        config = self.get_input_config()
        output_dir = self.get_output_dir()

        if self.variable == '':
            fitFolderName = f'runFits_mu_fiducial'
        else:
            fitFolderName = f'runFits_{self.variable}'

        output = []
        if self.variable == '':
            cat = "r"
            output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'impact', 'unblinded', 'impacts')]
            output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'impact', 'unblinded', 'impacts', f'impacts_unblinded.pdf')]
            output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'impact', 'unblinded', 'impacts', 'impacts_corrected_dropBkgModelParams.json')]
            
        else:
            for cat in combineVariableDict(self.variable, self.year)['paramStrNoOne']:
                output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'impact', 'unblinded', 'impacts')]
                output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'impact', 'unblinded', 'impacts', f'impacts_unblinded_{cat}.pdf')]
                output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'impact', 'unblinded', 'impacts', 'impacts_corrected_dropBkgModelParams.json')]
                
        output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'impact', 'unblinded', 'impacts', f'impacts.json')]
        
        outputFileTargets = []
                
        for _, current_output_path in enumerate(output):
            outputFileTargets.append(law.LocalFileTarget(current_output_path))
            
        # print(outputFileTargets)

        return outputFileTargets

    def run(self):
       
        if self.variable == '':
            fitFolderName = f'runFits_mu_fiducial'
        else:
            fitFolderName = f'runFits_{self.variable}'
                  
        config = self.get_input_config()
        output_dir = self.get_output_dir() 
            
        if self.variable == '':
            datacard_path = os.path.join(output_dir, 'Combine', outdir, f'Datacard_{self.year}.root')
        else:
            datacard_path = os.path.join(output_dir, 'Combine', outdir, f'Datacard_{self.variable}_{self.year}.root')

        cwd = os.getcwd()

        if self.batch_flavor == "slurm/psi":
            # Have to use /scratch/batch_username/ for slurm/psi
            if "/work" in output_dir:
                execute_command([f'mkdir -p {output_dir}/Combine/{fitFolderName}/impact/unblinded/impacts'], shell=True)
            else:   
                execute_command([f'xrdfs root://t3dcachedb03.psi.ch:1094/ mkdir -p {output_dir}/Combine/{fitFolderName}/impact/unblinded/impacts'], shell=True)

            os.environ["TARGET_PATH"] = f"/scratch/{os.environ['USER']}/{os.environ['SLURM_JOB_ID']}"
            execute_command([f'mkdir -p $TARGET_PATH/Combine/{fitFolderName}/impact/unblinded/impacts'], shell=True)
            os.chdir(os.path.join(os.environ["TARGET_PATH"], 'Combine', fitFolderName, 'impact', 'unblinded'))

            if "/work" in output_dir:
                slurm_copy_command = [
                    'cp', '-rf',
                    f'{output_dir}/Combine/{fitFolderName}/impact/unblinded',
                    f'{os.environ["TARGET_PATH"]}/Combine/{fitFolderName}/impact'
                ]
            else:
                slurm_copy_command = [
                    'xrdcp', '-rf',
                    'root://t3dcachedb03.psi.ch:1094//'+ f'{output_dir}/Combine/{fitFolderName}/impact/unblinded',
                    f'{os.environ["TARGET_PATH"]}/Combine/{fitFolderName}/impact'
                ]
            execute_command(slurm_copy_command)
            temp_output_dir = os.environ["TARGET_PATH"]
        else:
            execute_command([f'mkdir -p {output_dir}/Combine/{fitFolderName}/impact/unblinded/impacts'], shell=True)
            os.chdir(os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'impact', 'unblinded'))
            temp_output_dir = output_dir

        if self.variable == '':
            arguments = [
                "combineTool.py",
                "-M", "Impacts",
                "-d", datacard_path,
                "-m", f"{HIGGS_MASS}",
                "-o", "impacts/impacts.json"
            ]
            command = arguments
            # print(command)
            try:
                result = subprocess.run(command, check=True, text=True, capture_output=True)
                print("Script output:", result.stdout)
                print("Script executed successfully.")
            except subprocess.CalledProcessError as e:
                print("Error executing script:", e.stderr)
                
            arguments = [
                "python3",
                f"{os.path.join(os.environ['ANALYSIS_PATH'], 'Plots', 'correctImpacts.py')}",
                "--impactsJson", f"{os.path.join(temp_output_dir, 'Combine', outdir, fitFolderName, 'impact', 'unblinded', 'impacts', 'impacts.json')}",
                "--frozenParam", "MH",
                "--dropBkgModelParams"
            ]
            command = arguments
            # print(command)
            try:
                result = subprocess.run(command, check=True, text=True, capture_output=True)
                print("Script output:", result.stdout)
                print("Script executed successfully.")
            except subprocess.CalledProcessError as e:
                print("Error executing script:", e.stderr)
                
            arguments = [
                "plotImpacts.py",
                "-i", f"{os.path.join(temp_output_dir, 'Combine', outdir, fitFolderName, 'impact', 'unblinded', 'impacts', 'impacts_corrected_dropBkgModelParams.json')}",
                "-o", "impacts/impacts_unblinded",
            ]
            if config['combine_impacts']['not_show_POI']:
                arguments.append("--blind")
            command = arguments
            # print(command)
            try:
                result = subprocess.run(command, check=True, text=True, capture_output=True)
                print("Script output:", result.stdout)
                print("Script executed successfully.")
            except subprocess.CalledProcessError as e:
                print("Error executing script:", e.stderr)
        else:
            arguments = [
                "combineTool.py",
                "-M", "Impacts",
                "-d", datacard_path,
                "--freezeParameters", "MH",
                "-m", f"{HIGGS_MASS}",
                "-o", "impacts/impacts.json",
            ]
            if (config["combine_impacts"]["exclude"] != ""):
                arguments.append("--exclude")
                arguments.append(config["combine_impacts"]["exclude"])
            command = arguments
            # print(command)
            try:
                result = subprocess.run(command, check=True, text=True, capture_output=True)
                print("Script output:", result.stdout)
                print("Script executed successfully.")
            except subprocess.CalledProcessError as e:
                print("Error executing script:", e.stderr)
                
            arguments = [
                "python3",
                f"{os.path.join(os.environ['ANALYSIS_PATH'], 'Plots', 'correctImpacts.py')}",
                "--impactsJson", f"{os.path.join(temp_output_dir, 'Combine', outdir, fitFolderName, 'impact', 'unblinded', 'impacts', 'impacts.json')}",
                "--frozenParam", "MH",
                "--dropBkgModelParams"
            ]
            if (config["combine_impacts"]["exclude"] != ""):
                arguments.append("--exclude")
                arguments.append(config["combine_impacts"]["exclude"])
            command = arguments
            # print(command)
            try:
                result = subprocess.run(command, check=True, text=True, capture_output=True)
                print("Script output:", result.stdout)
                print("Script executed successfully.")
            except subprocess.CalledProcessError as e:
                print("Error executing script:", e.stderr)
                
            for cat in combineVariableDict(self.variable, self.year)['paramStrNoOne']:
                arguments = [
                    "plotImpacts.py",
                    "-i", f"{os.path.join(temp_output_dir, 'Combine', outdir, fitFolderName, 'impact', 'unblinded', 'impacts', 'impacts_corrected_dropBkgModelParams.json')}",
                    "-o", f"impacts/impacts_unblinded_{cat}",
                    "--POI", f"{cat}",
                    "--translate", f"{os.path.join(os.environ['ANALYSIS_PATH'], 'Combine', 'pois.json')}",
                ]
                if config['combine_impacts']['not_show_POI']:
                    arguments.append("--blind")
                command = arguments
                # print(command)
                try:
                    result = subprocess.run(command, check=True, text=True, capture_output=True)
                    print("Script output:", result.stdout)
                    print("Script executed successfully.")
                except subprocess.CalledProcessError as e:
                    print("Error executing script:", e.stderr)

        # Copy the files back to pnfs if we are on slurm/psi
        if self.batch_flavor == "slurm/psi":
            # Have to copy over the output to the final directory
            # Don't forget to VOMS!
            if "/work" in output_dir:
                slurm_copy_command = [
                    'cp', '-rf',
                    f"{os.environ['TARGET_PATH']}/Combine/",
                    output_dir
                ]
            else:
                slurm_copy_command = [
                    'xrdcp', '-rf',
                    f"{os.environ['TARGET_PATH']}/Combine/",
                    'root://t3dcachedb03.psi.ch:1094//'+output_dir
                ]
            print(slurm_copy_command)
            execute_command(slurm_copy_command)
            # Clean up the temporary directory
            shutil.rmtree(os.environ["TARGET_PATH"])
            
        os.chdir(cwd)
        
        
class MggBestFit(Task, law.LocalWorkflow, HTCondorWorkflow, SlurmWorkflow): #(law.Task): #(Task, HTCondorWorkflow, law.LocalWorkflow):
    output_dir = law.Parameter(default = '', description="Path to the output directory")
    variable = law.Parameter(default="", description="Variable to be used")
    year = law.Parameter(default='2022', description="Year")

    batch_flavor = law.Parameter(default="htcondor", description="Batch system to use")

    # def requires(self):
    def workflow_requires(self):
        workflow_reqs = super().workflow_requires()

        tasks = {}

        if workflow_reqs:
            tasks.update(workflow_reqs)
        
        config = self.get_input_config()
        output_dir = self.get_output_dir()

        toyConfig = config["combine_mggToys"]

        tasks["RunT2WS"] = RunText2Workspace(output_dir=output_dir, variable=self.variable, year=self.year, batch_flavor=self.batch_flavor, workflow=toyConfig["execution"], version=self.variable if self.variable != "" else "inclusive", slurm_partition=toyConfig.get('batchPartition', None), slurm_memory=toyConfig.get('batchMemory', None), slurm_max_runtime=toyConfig.get('batchMaxRuntime', None), htcondor_partition=toyConfig.get('batchPartition', None), htcondor_memory=toyConfig.get('batchMemory', None), htcondor_max_runtime=toyConfig.get('batchMaxRuntime', None))

        return tasks

    def create_branch_map(self):
            
        if self.variable in ["", "tuto"]:
            cat_list = ["r"]
        else:
            cat_list = combineVariableDict(self.variable, self.year)['paramStrNoOne']

        branch_map = {i: current_cat for i, current_cat in enumerate(cat_list)}
        return branch_map

    def output(self):        
        cat = self.branch_data
                
        config = self.get_input_config()
        output_dir = self.get_output_dir()

        if self.variable == '':
            fitFolderName = f'runFits_mu_fiducial'
        else:
            fitFolderName = f'runFits_{self.variable}'
            
        output = [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'postFit')]
        output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'postFit', f'SplusBModels_{cat}')]
        output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'postFit', f'SplusBModels_{cat}', f'higgsCombine_bestfit_syst_obs_{cat}.MultiDimFit.mH{HIGGS_MASS}.root')]
        
        outputFileTargets = []
                
        for _, current_output_path in enumerate(output):
            outputFileTargets.append(law.LocalFileTarget(current_output_path))
            
        # print(outputFileTargets)

        return outputFileTargets

    def run(self):
        cat = self.branch_data
       
        if self.variable == '':
            fitFolderName = f'runFits_mu_fiducial'
            
        else:
            fitFolderName = f'runFits_{self.variable}'
                  
        config = self.get_input_config()
        output_dir = self.get_output_dir() 
            
        if self.variable == '':
            datacard_path = os.path.join(output_dir, 'Combine', outdir, f'Datacard_{self.year}.root')
        else:
            datacard_path = os.path.join(output_dir, 'Combine', outdir, f'Datacard_{self.variable}_{self.year}.root')
             
        cwd = os.getcwd()

        if self.batch_flavor == "slurm/psi":
            # Have to use /scratch/batch_username/ for slurm/psi
            if "/work" in output_dir:
                execute_command([f'mkdir -p {output_dir}/Combine/{fitFolderName}/postFit/SplusBModels_{cat}'], shell=True)
            else:   
                execute_command([f'xrdfs root://t3dcachedb03.psi.ch:1094/ mkdir -p {output_dir}/Combine/{fitFolderName}/postFit/SplusBModels_{cat}'], shell=True)

            os.environ["TARGET_PATH"] = f"/scratch/{os.environ['USER']}/{os.environ['SLURM_JOB_ID']}"
            execute_command([f'mkdir -p $TARGET_PATH/Combine/{fitFolderName}/postFit/SplusBModels_{cat}'], shell=True)
            os.chdir(os.path.join(os.environ["TARGET_PATH"], 'Combine', fitFolderName, 'postFit', f'SplusBModels_{cat}'))
        else:
            execute_command([f'mkdir -p {output_dir}/Combine/{fitFolderName}/postFit/SplusBModels_{cat}'], shell=True)
            os.chdir(os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'postFit', f'SplusBModels_{cat}'))

        arguments = [
            "combine",
            "--floatOtherPOIs", "1",
            "-P", f"{cat}",
            "--freezeParameters", "MH",
            "--saveInactivePOI", "1",
            "--saveWorkspace",
            "--saveSpecifiedNuis", "all",
            "--cminDefaultMinimizerStrategy", "0",
            "--X-rtd", "MINIMIZER_freezeDisassociatedParams",
            "--X-rtd", "MINIMIZER_multiMin_hideConstants",
            "--X-rtd", "MINIMIZER_multiMin_maskConstraints",
            "--X-rtd", "MINIMIZER_multiMin_maskChannels=2",
            "-M", f"{config['combine_mggToys']['loadSnapshot']}",
            "-m", f"{HIGGS_MASS}",
            "-d", datacard_path,
            "-n", f"_bestfit_syst_obs_{cat}"
        ]
        command = arguments
        # print(command)
        try:
            result = subprocess.run(command, check=True, text=True, capture_output=True)
            print("Script output:", result.stdout)
            print("Script executed successfully.")
        except subprocess.CalledProcessError as e:
            print("Error executing script:", e.stderr)

        # Copy the files back to pnfs if we are on slurm/psi
        if self.batch_flavor == "slurm/psi":
            # Have to copy over the output to the final directory
            # Don't forget to VOMS!
            if "/work" in output_dir:
                slurm_copy_command = [
                    'cp', '-rf',
                    f"{os.environ['TARGET_PATH']}/Combine/",
                    output_dir
                ]
            else:
                slurm_copy_command = [
                    'xrdcp', '-rf',
                    f"{os.environ['TARGET_PATH']}/Combine/",
                    'root://t3dcachedb03.psi.ch:1094//'+output_dir
                ]
            print(slurm_copy_command)
            execute_command(slurm_copy_command)
            # Clean up the temporary directory
            shutil.rmtree(os.environ["TARGET_PATH"])
        
        os.chdir(cwd)
        
class MggToyGeneration(Task, law.LocalWorkflow, HTCondorWorkflow, SlurmWorkflow): #(law.Task): #(Task, HTCondorWorkflow, law.LocalWorkflow):
    is_postfit = law.Parameter(default=False, description="Flag that signifies if toys are created for postfit mass distributions.")


    # def requires(self):
    def workflow_requires(self):
        workflow_reqs = super().workflow_requires()

        tasks = {}

        if workflow_reqs:
            tasks.update(workflow_reqs)
        
        config = self.get_input_config()

        mggConfig = config['combine_mggToys']
        
        output_dir = self.get_output_dir()

        if convert_boolean_string(self.is_postfit):
            tasks["MggBestFit"] = MggBestFit.req(
                self,
                output_dir=output_dir, 
                workflow=mggConfig['execution'], 
                slurm_partition=mggConfig['batchPartition'], 
                slurm_memory=mggConfig['batchMemory'], 
                slurm_max_runtime=mggConfig['batchMaxRuntime'], 
                htcondor_partition=mggConfig['batchPartition'], 
                htcondor_memory=mggConfig['batchMemory'], 
                htcondor_max_runtime=mggConfig['batchMaxRuntime'])
        else:
            tasks["RunT2WS"] = RunText2Workspace.req(
                self,
                output_dir=output_dir, 
                years=self.year,
                )
            
        return tasks

    def create_branch_map(self):
        

                  
        #Load central config file
        config = self.get_input_config()
            
        if self.variable in ['', 'tuto']:
            cat_list = ["r"]
        elif self.variable == "MH":
            cat_list = ["MH"]
        else:
            cat_list = combineVariableDict(self.variable, self.year)['paramStrNoOne']
        nToys = config['combine_mggToys']['nToys']
        
        toy_cat_list = [
            (toy, cat)
            for toy in range(int(nToys))
            for cat in cat_list
        ]
        
        branch_map = {i: current_toy for i, current_toy in enumerate(toy_cat_list)}
        return branch_map

    def output(self):  
        outdir = f'outdir_{self.year}_{self.variable}' if self.variable != '' else ''      
        toy, cat = self.branch_data
                
        output_dir = self.get_output_dir()

        if self.variable == '':
            fitFolderName = f'runFits_mu_fiducial'
        else:
            fitFolderName = f'runFits_{self.variable}'
            
        if convert_boolean_string(self.is_postfit):
            output = [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'postFit', f'SplusBModels_{cat}', 'toys')]
            output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'postFit', f'SplusBModels_{cat}', 'toys', 'filechecker')]
            output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'postFit', f'SplusBModels_{cat}', 'toys', 'filechecker', f'toy_{toy}_ok.txt')]
            output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'postFit', f'SplusBModels_{cat}', 'toys', f'toy_{toy}.root')]
        else:
            output = [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'preFit')]
            output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'preFit', f'SplusBModels_{cat}')]
            output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'preFit', f'SplusBModels_{cat}', 'toys')]
            output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'preFit', f'SplusBModels_{cat}', 'toys', 'filechecker')]
            output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'preFit', f'SplusBModels_{cat}', 'toys', 'filechecker', f'toy_{toy}_ok.txt')]
            output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'preFit', f'SplusBModels_{cat}', 'toys', f'toy_{toy}.root')]
        
        outputFileTargets = []
                
        for _, current_output_path in enumerate(output):
            outputFileTargets.append(law.LocalFileTarget(current_output_path))
            
        # print(outputFileTargets)

        return outputFileTargets

    def run(self):
        toy, cat = self.branch_data
       
        if self.variable == '':
            fitFolderName = f'runFits_mu_fiducial'
            
        else:
            fitFolderName = f'runFits_{self.variable}'
                  
        config = self.get_input_config()
        output_dir = self.get_output_dir() 

        outdir = f'outdir_{self.year}_{self.variable}' if self.variable != '' else ''
            
        if self.variable == '':
            datacard_path = os.path.join(output_dir, 'Combine', outdir, f'Datacard_{self.year}.root')
        else:
            datacard_path = os.path.join(output_dir, 'Combine', outdir, f'Datacard_{self.variable}_{self.year}.root')
             
        cwd = os.getcwd()

        if convert_boolean_string(self.is_postfit):

            os.environ["TARGET_PATH"] = f"/scratch/{os.environ['USER']}/{os.environ['SLURM_JOB_ID']}"

            if self.batch_flavor == "slurm/psi":
                # Have to use /scratch/batch_username/ for slurm/psi
                if "/work" in output_dir:
                    execute_command([f'mkdir -p {output_dir}/Combine/outdir_{self.year}_{self.variable}/{fitFolderName}/postFit/SplusBModels_{cat}/toys/filechecker'], shell=True)
                else:   
                    execute_command([f'xrdfs root://t3dcachedb03.psi.ch:1094/ mkdir -p {output_dir}/Combine/{fitFolderName}/postFit/SplusBModels_{cat}/toys/filechecker'], shell=True)

                execute_command([f'mkdir -p $TARGET_PATH/Combine/{fitFolderName}/postFit/SplusBModels_{cat}/toys/filechecker'], shell=True)
                os.chdir(os.path.join(os.environ["TARGET_PATH"], 'Combine', fitFolderName, 'postFit', f'SplusBModels_{cat}', 'toys'))
            else:
                execute_command([f'mkdir -p {output_dir}/Combine/{outdir}/{fitFolderName}/postFit/SplusBModels_{cat}/toys/filechecker'], shell=True)
                os.chdir(os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'postFit', f'SplusBModels_{cat}', 'toys'))
        
            best_fit = os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'postFit', f'SplusBModels_{cat}', f'higgsCombine_bestfit_syst_obs_{cat}.MultiDimFit.mH{HIGGS_MASS}.root')

            f = ROOT.TFile(best_fit)
            w = f.Get("w")
            w.loadSnapshot(config['combine_mggToys']['loadSnapshot'])
            poi_bf = w.var(cat).getVal()

            arguments = [
                "combine",
                best_fit,
                "-M", "GenerateOnly",
                "-m", f"{HIGGS_MASS}",
                "--saveWorkspace",
                "--toysFrequentist",
                "--bypassFrequentistFit",
                "-t", "1",
                "-s", "-1",
                "-n", f"_{toy}_gen_step",
                "--setParameters", f"{cat}={poi_bf}",
                "--snapshotName", f"{config['combine_mggToys']['loadSnapshot']}"
            ]
            command = arguments
            # print(command)
            try:
                result = subprocess.run(command, check=True, text=True, capture_output=True)
                print("Script output:", result.stdout)
                print("Script executed successfully.")
            except subprocess.CalledProcessError as e:
                print("Error executing script:", e.stderr)
                
            # Define the source pattern and destination path
            # On the PSI Tier 3 when executed with SLURM, the Toys are on the Storage Element which needs special handling
            # For this reason, introduce a "toy_output_dir" which is in the slurm/psi case the /scratch directory
            # and in the other cases the output_dir
            if self.batch_flavor == "slurm/psi" and "/pnfs" in output_dir:
                toy_output_dir = os.environ["TARGET_PATH"]
            else:
                toy_output_dir = output_dir

            source_pattern = os.path.join(toy_output_dir, 'Combine', outdir, fitFolderName, 'postFit', f'SplusBModels_{cat}', 'toys', f'higgsCombine_{toy}_gen_step*.root')
            destination = os.path.join(toy_output_dir, 'Combine', outdir, fitFolderName, 'postFit', f'SplusBModels_{cat}', 'toys', f'gen_{toy}.root')

            # Use glob to find files matching the source pattern
            source_files = glob.glob(source_pattern)

            if not source_files:
                print("No files found matching the pattern.")
            else:
                # Move each matched file to the destination
                for source_file in source_files:
                    try:
                        print(f"Moving {source_file} to {destination}")
                        if self.batch_flavor == "slurm/psi" and (("/pnfs" in source_file) or ("/pnfs" in destination)):
                            manually_move_t3(source_file, destination)
                        else:
                            shutil.move(source_file, destination)
                        print("File moved successfully.")
                    except Exception as e:
                        print(f"Error moving file {source_file}: {e}")

            arguments = [
                "combine",
                f"gen_{toy}.root",
                "-m", f"{HIGGS_MASS}",
                "-M", f"{config['combine_mggToys']['loadSnapshot']}",
                "-P", f"{cat}",
                "--floatOtherPOIs=1",
                "--saveWorkspace",
                "--toysFrequentist",
                "--bypassFrequentistFit",
                "-t", "1",
                "--setParameters", f"{cat}={poi_bf}",
                "-s", "-1",
                "-n", f"_{toy}_fit_step",
                "--cminDefaultMinimizerStrategy", "0",
                "--X-rtd", "MINIMIZER_freezeDisassociatedParams",
                "--X-rtd", "MINIMIZER_multiMin_hideConstants",
                "--X-rtd", "MINIMIZER_multiMin_maskConstraints",
                "--X-rtd", "MINIMIZER_multiMin_maskChannels=2"
            ]
            command = arguments
            # print(command)
            try:
                result = subprocess.run(command, check=True, text=True, capture_output=True)
                print("Script output:", result.stdout)
                print("Script executed successfully.")
            except subprocess.CalledProcessError as e:
                print("Error executing script:", e.stderr)    
                
            # Define the source pattern and destination path
            source_pattern = os.path.join(toy_output_dir, 'Combine', outdir, fitFolderName, 'postFit', f'SplusBModels_{cat}', 'toys', f'higgsCombine_{toy}_fit_step*.root')
            destination = os.path.join(toy_output_dir, 'Combine', outdir, fitFolderName, 'postFit', f'SplusBModels_{cat}', 'toys', f'fit_{toy}.root')

            # Use glob to find files matching the source pattern
            source_files = glob.glob(source_pattern)

            if not source_files:
                print("No files found matching the pattern.")
            else:
                # Move each matched file to the destination
                for source_file in source_files:
                    try:
                        print(f"Moving {source_file} to {destination}")
                        if self.batch_flavor == "slurm/psi" and (("/pnfs" in source_file) or ("/pnfs" in destination)):
                            manually_move_t3(source_file, destination)
                        else:
                            shutil.move(source_file, destination)
                        print("File moved successfully.")
                    except Exception as e:
                        print(f"Error moving file {source_file}: {e}")

            arguments = [
                "combine",
                f"fit_{toy}.root",
                "-m", f"{HIGGS_MASS}",
                "--snapshotName", f"{config['combine_mggToys']['loadSnapshot']}",
                "-M", "GenerateOnly",
                "--saveToys",
                "--toysFrequentist",
                "--bypassFrequentistFit",
                "-t", "-1",
                "-n", f"_{toy}_throw_step"
            ]
            if self.variable == '':
                arguments.append("--setParameters")
                arguments.append("r=0")
            else:
                arguments.append("--setParameters")
                arguments.append(f"{cat}=0")
            command = arguments
            # print(command)
            try:
                result = subprocess.run(command, check=True, text=True, capture_output=True)
                print("Script output:", result.stdout)
                print("Script executed successfully.")
            except subprocess.CalledProcessError as e:
                print("Error executing script:", e.stderr)
                
            # Define the source pattern and destination path
            source_pattern = os.path.join(toy_output_dir, 'Combine', outdir, fitFolderName, 'postFit', f'SplusBModels_{cat}', 'toys', f'higgsCombine_{toy}_throw_step*.root')
            destination = os.path.join(toy_output_dir, 'Combine', outdir, fitFolderName, 'postFit', f'SplusBModels_{cat}', 'toys', f'toy_{toy}.root')

            # Use glob to find files matching the source pattern
            source_files = glob.glob(source_pattern)

            if not source_files:
                print("No files found matching the pattern.")
            else:
                # Move each matched file to the destination
                for source_file in source_files:
                    try:
                        print(f"Moving {source_file} to {destination}")
                        if self.batch_flavor == "slurm/psi" and (("/pnfs" in source_file) or ("/pnfs" in destination)):
                            manually_move_t3(source_file, destination)
                        else:
                            shutil.move(source_file, destination)
                        print("File moved successfully.")
                    except Exception as e:
                        print(f"Error moving file {source_file}: {e}")
                        
            # Define the files to remove
            files_to_remove = [os.path.join(toy_output_dir, 'Combine', outdir, fitFolderName, 'postFit', f'SplusBModels_{cat}', 'toys', f'gen_{toy}.root'), os.path.join(toy_output_dir, 'Combine', outdir, fitFolderName, 'postFit', f'SplusBModels_{cat}', 'toys', f'fit_{toy}.root')]

            # Remove each specified file
            for file_path in files_to_remove:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        print(f"{file_path} removed successfully.")
                    else:
                        print(f"{file_path} does not exist.")
                except Exception as e:
                    print(f"Error removing file {file_path}: {e}")
            
            # Check if toy is > 1000 bytes (== file empty)
            try:
                file_size = os.path.getsize(os.path.join(toy_output_dir, 'Combine', outdir, fitFolderName, 'postFit', f'SplusBModels_{cat}', 'toys', f'toy_{toy}.root'))  # Get the file size in bytes
                if file_size > 1000:
                    with open(os.path.join(toy_output_dir, 'Combine', outdir, fitFolderName, 'postFit', f'SplusBModels_{cat}', 'toys', 'filechecker', f'toy_{toy}_ok.txt'), 'w') as f:
                        pass
            except OSError:
                # Handle the case where the file does not exist or is inaccessible
                print(f"Error creating file. Probably I/O error.")
                return False

        else:

            if self.batch_flavor == "slurm/psi":
                # Have to use /scratch/batch_username/ for slurm/psi
                if "/work" in output_dir:
                    execute_command([f'mkdir -p {output_dir}/Combine/{fitFolderName}/preFit/SplusBModels_{cat}/toys/filechecker'], shell=True)
                else:   
                    execute_command([f'xrdfs root://t3dcachedb03.psi.ch:1094/ mkdir -p {output_dir}/Combine/{fitFolderName}/preFit/SplusBModels_{cat}/toys/filechecker'], shell=True)

                os.environ["TARGET_PATH"] = f"/scratch/{os.environ['USER']}/{os.environ['SLURM_JOB_ID']}"
                execute_command([f'mkdir -p $TARGET_PATH/Combine/{fitFolderName}/preFit/SplusBModels_{cat}/toys/filechecker'], shell=True)
                os.chdir(os.path.join(os.environ["TARGET_PATH"], 'Combine', fitFolderName, 'preFit', f'SplusBModels_{cat}', 'toys'))
            else:
                execute_command([f'mkdir -p {output_dir}/Combine/{outdir}/{fitFolderName}/preFit/SplusBModels_{cat}/toys/filechecker'], shell=True)
                os.chdir(os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'preFit', f'SplusBModels_{cat}', 'toys'))

            arguments = [
                "combine",
                "-M", "GenerateOnly",
                "-d", datacard_path,
                "-m", f"{HIGGS_MASS}",
                "--saveWorkspace",
                "--toysFrequentist",
                "--bypassFrequentistFit",
                "-t", "1",
                "-s", "-1",
                "-n", f"_{toy}_gen_step",
            ]
            arguments.append("--setParameters")
            if self.variable in ['', 'tuto', 'MH']:
                arguments.append('r=1')
            else:
                arguments.append(f"""{",".join(combineVariableDict(self.variable, self.year)['paramStr'])}""")
            command = arguments
            print(' '.join(command))
            try:
                result = subprocess.run(command, check=True, text=True, capture_output=True)
                print("Script output:", result.stdout)
                print("Script executed successfully.")
            except subprocess.CalledProcessError as e:
                print("Error executing script:", e.stderr)

            # Define the source pattern and destination path
            # On the PSI Tier 3 when executed with SLURM, the Toys are on the Storage Element which needs special handling
            # For this reason, introduce a "toy_output_dir" which is in the slurm/psi case the /scratch directory
            # and in the other cases the output_dir
            if self.batch_flavor == "slurm/psi" and "/pnfs" in output_dir:
                toy_output_dir = os.environ["TARGET_PATH"]
            else:
                toy_output_dir = output_dir
                
            source_pattern = os.path.join(toy_output_dir, 'Combine', outdir, fitFolderName, 'preFit', f'SplusBModels_{cat}', 'toys', f'higgsCombine_{toy}_gen_step*.root')
            destination = os.path.join(toy_output_dir, 'Combine', outdir, fitFolderName, 'preFit', f'SplusBModels_{cat}', 'toys', f'gen_{toy}.root')

            # Use glob to find files matching the source pattern
            source_files = glob.glob(source_pattern)

            if not source_files:
                print("No files found matching the pattern.")
            else:
                # Move each matched file to the destination
                for source_file in source_files:
                    try:
                        print(f"Moving {source_file} to {destination}")
                        if (self.batch_flavor == "slurm/psi") and (("/pnfs" in source_file) or ("/pnfs" in destination)):
                            manually_move_t3(source_file, destination)
                        else:
                            shutil.move(source_file, destination)
                        print("File moved successfully.")
                    except Exception as e:
                        print(f"Error moving file {source_file}: {e}")

            arguments = [
                "combine",
                f"gen_{toy}.root",
                "-m", f"{HIGGS_MASS}",
                "-M", f"{config['combine_mggToys']['loadSnapshot']}",
                "-P", f"{cat}",
                "--floatOtherPOIs=1",
                "--saveWorkspace",
                "--toysFrequentist",
                "--bypassFrequentistFit",
                "-t", "1",
                "-s", "-1",
                "-n", f"_{toy}_fit_step",
                "--cminDefaultMinimizerStrategy", "0",
                "--X-rtd", "MINIMIZER_freezeDisassociatedParams",
                "--X-rtd", "MINIMIZER_multiMin_hideConstants",
                "--X-rtd", "MINIMIZER_multiMin_maskConstraints",
                "--X-rtd", "MINIMIZER_multiMin_maskChannels=2"
            ]
            arguments.append("--setParameters")
            if self.variable in ['', 'tuto', 'MH']:
                arguments.append('r=1')
            else:
                arguments.append(f"""{",".join(combineVariableDict(self.variable, self.year)['paramStr'])}""")
            command = arguments
            print(' '.join(command))
            try:
                result = subprocess.run(command, check=True, text=True, capture_output=True)
                print("Script output:", result.stdout)
                print("Script executed successfully.")
            except subprocess.CalledProcessError as e:
                print("Error executing script:", e.stderr)    
                
            # Define the source pattern and destination path
            source_pattern = os.path.join(toy_output_dir, 'Combine', outdir, fitFolderName, 'preFit', f'SplusBModels_{cat}', 'toys', f'higgsCombine_{toy}_fit_step*.root')
            destination = os.path.join(toy_output_dir, 'Combine', outdir, fitFolderName, 'preFit', f'SplusBModels_{cat}', 'toys', f'fit_{toy}.root')

            # Use glob to find files matching the source pattern
            source_files = glob.glob(source_pattern)

            if not source_files:
                print("No files found matching the pattern.")
            else:
                # Move each matched file to the destination
                for source_file in source_files:
                    try:
                        print(f"Moving {source_file} to {destination}")
                        if self.batch_flavor == "slurm/psi" and (("/pnfs" in source_file) or ("/pnfs" in destination)):
                            manually_move_t3(source_file, destination)
                        else:
                            shutil.move(source_file, destination)
                        print("File moved successfully.")
                    except Exception as e:
                        print(f"Error moving file {source_file}: {e}")
                
            arguments = [
                "combine",
                f"fit_{toy}.root",
                "-m", f"{HIGGS_MASS}",
                "--snapshotName", f"{config['combine_mggToys']['loadSnapshot']}",
                "-M", "GenerateOnly",
                "--saveToys",
                "--toysFrequentist",
                "--bypassFrequentistFit",
                "-t", "-1",
                "-n", f"_{toy}_throw_step"
            ]
            arguments.append("--setParameters")
            if self.variable in ['', 'tuto', 'MH']:
                arguments.append('r=0')
            else:
                arguments.append(f"""{(",".join(combineVariableDict(self.variable, self.year)['paramStr'])).replace("=1", "=0")}""")
            command = arguments
            print(' '.join(command))
            try:
                result = subprocess.run(command, check=True, text=True, capture_output=True)
                print("Script output:", result.stdout)
                print("Script executed successfully.")
            except subprocess.CalledProcessError as e:
                print("Error executing script:", e.stderr)
                
            # Define the source pattern and destination path
            source_pattern = os.path.join(toy_output_dir, 'Combine', outdir, fitFolderName, 'preFit', f'SplusBModels_{cat}', 'toys', f'higgsCombine_{toy}_throw_step*.root')
            destination = os.path.join(toy_output_dir, 'Combine', outdir, fitFolderName, 'preFit', f'SplusBModels_{cat}', 'toys', f'toy_{toy}.root')

            # Use glob to find files matching the source pattern
            source_files = glob.glob(source_pattern)

            if not source_files:
                print("No files found matching the pattern.")
            else:
                # Move each matched file to the destination
                for source_file in source_files:
                    try:
                        print(f"Moving {source_file} to {destination}")
                        if self.batch_flavor == "slurm/psi" and (("/pnfs" in source_file) or ("/pnfs" in destination)):
                            manually_move_t3(source_file, destination)
                        else:
                            shutil.move(source_file, destination)
                        print("File moved successfully.")
                    except Exception as e:
                        print(f"Error moving file {source_file}: {e}")
                        
            # Define the files to remove
            files_to_remove = [os.path.join(toy_output_dir, 'Combine', outdir, fitFolderName, 'preFit', f'SplusBModels_{cat}', 'toys', f'gen_{toy}.root'), os.path.join(toy_output_dir, 'Combine', outdir, fitFolderName, 'preFit', f'SplusBModels_{cat}', 'toys', f'fit_{toy}.root')]

            # Remove each specified file
            for file_path in files_to_remove:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        print(f"{file_path} removed successfully.")
                    else:
                        print(f"{file_path} does not exist.")
                except Exception as e:
                    print(f"Error removing file {file_path}: {e}")
            
            # Check if toy is > 1000 bytes (== file empty)
            try:
                file_size = os.path.getsize(os.path.join(toy_output_dir, 'Combine', outdir, fitFolderName, 'preFit', f'SplusBModels_{cat}', 'toys', f'toy_{toy}.root'))  # Get the file size in bytes
                if file_size > 1000:
                    with open(os.path.join(toy_output_dir, 'Combine', outdir, fitFolderName, 'preFit', f'SplusBModels_{cat}', 'toys', 'filechecker', f'toy_{toy}_ok.txt'), 'w') as f:
                        pass
            except OSError:
                # Handle the case where the file does not exist or is inaccessible
                print(f"Error creating file. Probably I/O error.")
                return False

        # Copy the files back to pnfs if we are on slurm/psi
        if self.batch_flavor == "slurm/psi":
            # Have to copy over the output to the final directory
            # Don't forget to VOMS!
            if "/work" in output_dir:
                slurm_copy_command = [
                    'cp', '-rf',
                    f"{os.environ['TARGET_PATH']}/Combine/",
                    output_dir
                ]
            else:
                slurm_copy_command = [
                    'xrdcp', '-rf',
                    f"{os.environ['TARGET_PATH']}/Combine/",
                    'root://t3dcachedb03.psi.ch:1094//'+output_dir
                ]
            print(slurm_copy_command)
            execute_command(slurm_copy_command)
            # Clean up the temporary directory
            shutil.rmtree(os.environ["TARGET_PATH"])
        
        os.chdir(cwd)
        
class MggDistribution(MultiYearTask): #(law.Task): #(Task, HTCondorWorkflow, law.LocalWorkflow):
    is_postfit = law.Parameter(default=False, description="Flag that signifies if toys are created for postfit mass distributions.")


    # def requires(self):
    def _requires_single(self):
        
        tasks = {}
        
        config = self.get_input_config()
        output_dir = self.get_output_dir()
        
        mggConfig = config['combine_mggToys']
        
        if config['combine_mggToys']['doBands']:
            if convert_boolean_string(self.is_postfit):
                tasks["MggToyGeneration"] = MggToyGeneration.req(
                    self,
                    output_dir=output_dir, 
                    is_postfit=convert_boolean_string(self.is_postfit), 
                    workflow=mggConfig["execution"], 
                    slurm_partition=mggConfig.get('batchPartition', None), 
                    slurm_memory=mggConfig.get('batchMemory', None), 
                    slurm_max_runtime=mggConfig.get('batchMaxRuntime', None), 
                    htcondor_partition=mggConfig.get('batchPartition', None), 
                    htcondor_memory=mggConfig.get('batchMemory', None), 
                    htcondor_max_runtime=mggConfig.get('batchMaxRuntime', None))
            else:
                tasks["MggToyGeneration"] = MggToyGeneration.req(
                    self,
                    output_dir=output_dir, 
                    year=self.year,
                    is_postfit=convert_boolean_string(self.is_postfit), 
                    workflow=mggConfig["execution"], 
                    slurm_partition=mggConfig.get('batchPartition', None), 
                    slurm_memory=mggConfig.get('batchMemory', None), 
                    slurm_max_runtime=mggConfig.get('batchMaxRuntime', None), 
                    htcondor_partition=mggConfig.get('batchPartition', None), 
                    htcondor_memory=mggConfig.get('batchMemory', None), 
                    htcondor_max_runtime=mggConfig.get('batchMaxRuntime', None))
        else:
            if convert_boolean_string(self.is_postfit):
                fitConfig = config["combine_fit"]
                tasks["CreateUnblindedFit"] = CreateUnblindedFit(output_dir=output_dir, variable=self.variable, year=self.year, batch_flavor=self.batch_flavor, version=f"{self.variable if self.variable != '' else 'r'}_prefit", workflow=fitConfig["execution"], slurm_partition=fitConfig['batchPartition'], slurm_memory=fitConfig['batchMemory'], slurm_max_runtime=fitConfig['batchMaxRuntime'], htcondor_partition=fitConfig['batchPartition'], htcondor_memory=fitConfig['batchMemory'], htcondor_max_runtime=fitConfig['batchMaxRuntime'])
            else:
                tasks["RunT2WS"] = RunText2Workspace.req(
                    self,
                    output_dir=output_dir, 
                    )
        
        return tasks


    def output(self):

        outdir = f'outdir_{self.year}_{self.variable}' if self.variable != '' else f'outdir_{self.year}_r'
        cat = self.variable
        
        output_dir = self.get_output_dir()

        skipIndivCat = False
        if self.variable == 'MH':
            fitFolderName = 'runFits_MH'
            years = yearMap[self.year]
            if len(years) > 1:
                skipIndivCat = True
                reco_cats_with_bmw = ['all']
            else:
                reco_cats_with_bmw = ['BEST', 'MEDIUM', 'WORST']

        elif self.variable == '':
            fitFolderName = f'runFits_{self.variable}'
            reco_cats_with_bmw = ['cat0', 'cat1', 'cat2']
            if "_" in self.year:
                cats = []
                for y in self.year.split("_"):
                    y2 = y[-2:]
                    for c in reco_cats_with_bmw:
                        cats.append(f"Y{y2}_{c}")
                reco_cats_with_bmw = cats

        else:
            fitFolderName = f'runFits_{self.variable}'
            reco_cats_with_bmw = [element for element in combineVariableDict(self.variable, self.year)['catsStrWithBMW'] if "_".join(cat.split("_")[2:]) in element]
            if "_" in self.year:
                cats = []
                for y in self.year.split("_"):
                    y2 = y[-2:]
                    for c in reco_cats_with_bmw:
                        cats.append(f"Y{y2}_{c}")
                reco_cats_with_bmw = cats
        
        output = []
        if convert_boolean_string(self.is_postfit):
            output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'postFit', 'jsons')]
            output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'postFit', 'jsons', f'catsWeights_sospb_{cat}_CMS_hgg_mass.json')]
            
            output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'postFit', f'SplusBModels_{cat}', f'{cat}_{catWithBMW}_CMS_hgg_mass.pdf') for catWithBMW in reco_cats_with_bmw]
            output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'postFit', f'SplusBModels_{cat}', f'{cat}_{catWithBMW}_CMS_hgg_mass.png') for catWithBMW in reco_cats_with_bmw]
            output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'postFit', f'SplusBModels_{cat}', f'{cat}_all_CMS_hgg_mass.pdf')]
            output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'postFit', f'SplusBModels_{cat}', f'{cat}_all_CMS_hgg_mass.png')]
            output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'postFit', f'SplusBModels_{cat}', f'{cat}_wall_CMS_hgg_mass.pdf')]
            output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'postFit', f'SplusBModels_{cat}', f'{cat}_wall_CMS_hgg_mass.png')]
        else:
            output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'preFit', 'jsons')]
            output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'preFit', 'jsons', f'catsWeights_sospb_{cat}_CMS_hgg_mass.json')]
            
            output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'preFit', f'SplusBModels_{cat}', f'{cat}_{catWithBMW}_CMS_hgg_mass.pdf') for catWithBMW in reco_cats_with_bmw]
            output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'preFit', f'SplusBModels_{cat}', f'{cat}_{catWithBMW}_CMS_hgg_mass.png') for catWithBMW in reco_cats_with_bmw]
            output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'preFit', f'SplusBModels_{cat}', f'{cat}_all_CMS_hgg_mass.pdf')]
            output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'preFit', f'SplusBModels_{cat}', f'{cat}_all_CMS_hgg_mass.png')]
            output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'preFit', f'SplusBModels_{cat}', f'{cat}_wall_CMS_hgg_mass.pdf')]
            output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'preFit', f'SplusBModels_{cat}', f'{cat}_wall_CMS_hgg_mass.png')]
                
        
        outputFileTargets = []
                
        for _, current_output_path in enumerate(output):
            outputFileTargets.append(law.LocalFileTarget(current_output_path))
            
        # print(outputFileTargets)

        return {self.year: outputFileTargets}

    def run(self):
        cat = self.variable
       
        if self.variable == '':
            fitFolderName = f'runFits_mu_fiducial'

        else:
            fitFolderName = f'runFits_{self.variable}'

        #Load central config file
        output_dir = self.get_output_dir()  
        config = self.get_input_config()
        outdir = f'outdir_{self.year}_{self.variable}' if self.variable != '' else f'outdir_{self.year}_r'
            
        main_dir = os.getcwd()
        
        if convert_boolean_string(self.is_postfit):

            if self.batch_flavor == "slurm/psi":
                # Have to use /scratch/batch_username/ for slurm/psi
                if "/work" in output_dir:
                    execute_command([f'mkdir -p {output_dir}/Combine/{fitFolderName}/postFit/SplusBModels_{cat}/'], shell=True)
                else:   
                    execute_command([f'xrdfs root://t3dcachedb03.psi.ch:1094/ mkdir -p {output_dir}/Combine/{fitFolderName}/postFit/SplusBModels_{cat}/'], shell=True)

                os.environ["TARGET_PATH"] = f"/scratch/{os.environ['USER']}/{os.environ['SLURM_JOB_ID']}"
                execute_command([f'mkdir -p $TARGET_PATH/Combine/{fitFolderName}/postFit/SplusBModels_{cat}/'], shell=True)
                
                if "/work" in output_dir:
                    slurm_copy_command = [
                        'cp', '-rf',
                        f'{output_dir}/Combine/{fitFolderName}/postFit/SplusBModels_{cat}/toys',
                        f'{os.environ["TARGET_PATH"]}/Combine/{fitFolderName}/postFit/SplusBModels_{cat}/'
                    ]
                else:
                    slurm_copy_command = [
                        'xrdcp', '-rf',
                        'root://t3dcachedb03.psi.ch:1094//'+ f'{output_dir}/Combine/{fitFolderName}/postFit/SplusBModels_{cat}/toys',
                        f'{os.environ["TARGET_PATH"]}/Combine/{fitFolderName}/postFit/SplusBModels_{cat}/'
                    ]
                execute_command(slurm_copy_command)
                os.chdir(os.path.join(os.environ["TARGET_PATH"], 'Combine', fitFolderName, 'postFit'))
            else:
                execute_command([f'mkdir -p {output_dir}/Combine/{fitFolderName}/postFit/SplusBModels_{cat}/'], shell=True)
                os.chdir(os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'postFit'))
            
            best_fit = os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'postFit', f'SplusBModels_{cat}', f'higgsCombine_bestfit_syst_obs_{cat}.MultiDimFit.mH{HIGGS_MASS}.root')
            
            if self.variable == '':
                # firstStep_path = os.path.join(output_dir, 'Combine', outdir, f'Datacard_{self.year}.root')
                reco_cats_with_bmw = ['cat0', 'cat1', 'cat2']

                if "_" in self.year:
                    cats = []
                    for y in self.year.split("_"):
                        y2 = y[-2:]
                        for c in reco_cats_with_bmw:
                            cats.append(f"Y{y2}_{c}")
                    reco_cats_with_bmw = cats

            else:
                # firstStep_path = os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'dataFit', f'higgsCombineDataPostFitScanFit_{cat}.MultiDimFit.mH{HIGGS_MASS}.root')
                reco_cats_with_bmw = [element for element in combineVariableDict(self.variable, self.year)['catsStrWithBMW'] if "_".join(cat.split("_")[2:]) in element]
                if "_" in self.year:
                    cats = []
                    for y in self.year.split("_"):
                        y2 = y[-2:]
                        for c in reco_cats_with_bmw:
                            cats.append(f"Y{y2}_{c}")
                    reco_cats_with_bmw = cats
            arguments = [
                "python3",
                f"{os.path.join(os.environ['ANALYSIS_PATH'], 'Plots', 'makeSplusBModelPlot.py')}",
                "--inputWSFile", best_fit,
                "--loadSnapshot", f"{config['combine_mggToys']['loadSnapshot']}",
                "--cats", f"{','.join(reco_cats_with_bmw)}",
                "--lumiLabel", get_lumi_label(self.year),
                # "--isPreliminary",
                "--doZeroes",
                "--unblind",
                "--translateCats", f"{os.path.join(os.environ['ANALYSIS_PATH'], 'Plots', 'cats.json')}",
                "--doSumCategories",
                # merge per year cat together
                # "--skipIndividualCatPlots",
                # "--cats Y22_BEST,Y23_BEST,Y24_BEST,Y25_BEST" 
                "--doCatWeights",
                "--saveWeights",
                "--ext", f"_{cat}",
                "--POI", f"{cat}"
            ]
            if config['combine_mggToys']['doBands']:
                arguments.append("--doBands")
                arguments.append("--doToyVeto")
                arguments.append("--saveToyYields")
            command = arguments
            print(' '.join(command))
            try:
                result = subprocess.run(command, check=True, text=True, capture_output=True)
                print("Script output:", result.stdout)
                print("Script executed successfully.")
            except subprocess.CalledProcessError as e:
                print("Error executing script:", e.stderr)
            
            # Change directory
            # os.chdir(f"./SplusBModels_{cat}")

            # if self.variable != "":
            #     # Extract parts from the parameter
            #     parts = cat.split('_')
            #     pattern = f"{parts[2]}_{parts[3]}"

            #     # Define source and target directories
            #     source_dir = "."
            #     target_dir = "../Plots"

            #     # Iterate over files in the source directory
            #     for filename in os.listdir(source_dir):
            #         # Check if the pattern is in the filename
            #         if pattern in filename:
            #             # Construct full source and destination paths
            #             source_path = os.path.join(source_dir, filename)
            #             target_path = os.path.join(target_dir, filename)
            #             # Copy the file to the target directory
            #             shutil.copy(source_path, target_path)
            #             print(f"Copied {filename} to {target_dir}")

            # Go back one directory
            os.chdir("..")
            
        else: 
            
            if self.batch_flavor == "slurm/psi":
                # Have to use /scratch/batch_username/ for slurm/psi
                if "/work" in output_dir:
                    execute_command([f'mkdir -p {output_dir}/Combine/{fitFolderName}/preFit/SplusBModels_{cat}/'], shell=True)
                else:   
                    execute_command([f'xrdfs root://t3dcachedb03.psi.ch:1094/ mkdir -p {output_dir}/Combine/{fitFolderName}/preFit/SplusBModels_{cat}/'], shell=True)
                os.environ["TARGET_PATH"] = f"/scratch/{os.environ['USER']}/{os.environ['SLURM_JOB_ID']}"
                execute_command([f'mkdir -p $TARGET_PATH/Combine/{fitFolderName}/preFit/SplusBModels_{cat}/'], shell=True)

                if "/work" in output_dir:
                    slurm_copy_command = [
                        'cp', '-rf',
                        f'{output_dir}/Combine/{fitFolderName}/preFit/SplusBModels_{cat}/toys',
                        f'{os.environ["TARGET_PATH"]}/Combine/{fitFolderName}/preFit/SplusBModels_{cat}/'
                    ]
                else:
                    slurm_copy_command = [
                        'xrdcp', '-rf',
                        'root://t3dcachedb03.psi.ch:1094//'+ f'{output_dir}/Combine/{fitFolderName}/preFit/SplusBModels_{cat}/toys',
                        f'{os.environ["TARGET_PATH"]}/Combine/{fitFolderName}/preFit/SplusBModels_{cat}/'
                    ]
                execute_command(slurm_copy_command)
                os.chdir(os.path.join(os.environ["TARGET_PATH"], 'Combine', fitFolderName, 'preFit'))
            else:
                execute_command([f'mkdir -p {output_dir}/Combine/{fitFolderName}/preFit/SplusBModels_{cat}/'], shell=True)
                os.chdir(os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'preFit'))
            
            if self.variable == '':
                datacard_path = os.path.join(output_dir, 'Combine', outdir, f'Datacard_{self.year}.root')
            else:
                datacard_path = os.path.join(output_dir, 'Combine', outdir, f'Datacard_{self.variable}_{self.year}.root')

            skipIndivCat = False
            if self.variable == '':
                reco_cats_with_bmw = ['cat0', 'cat1', 'cat2']

                if "_" in self.year:
                    cats = []
                    for y in self.year.split("_"):
                        y2 = y[-2:]
                        for c in reco_cats_with_bmw:
                            cats.append(f"Y{y2}_{c}")
                    reco_cats_with_bmw = cats

            elif self.variable == 'tuto':
                reco_cats_with_bmw = ['EBEB_highR9highR9', 'EBEB_highR9lowR9', 'EBEB_lowR9highR9', 'EBEE_highR9highR9', 'EBEE_highR9lowR9', 'EBEE_lowR9highR9', 'EEEB_highR9highR9', 'EEEB_highR9lowR9', 'EEEB_lowR9highR9', 'EEEE_incl']

            elif self.variable == 'MH':
                fitFolderName = 'runFits_MH'
                years = yearMap[self.year]
                if len(years) > 1:
                    skipIndivCat = True
                    reco_cats_with_bmw = ['all']
                else:
                    reco_cats_with_bmw = ['BEST', 'MEDIUM', 'WORST']
            else:
                reco_cats_with_bmw = [element for element in combineVariableDict(self.variable, self.year)['catsStrWithBMW'] if "_".join(cat.split("_")[2:]) in element]

                if "_" in self.year:
                    cats = []
                    for y in self.year.split("_"):
                        y2 = y[-2:]
                        for c in reco_cats_with_bmw:
                            cats.append(f"Y{y2}_{c}")
                    reco_cats_with_bmw = cats

            arguments = [
                "python3",
                f"{os.path.join(os.environ['ANALYSIS_PATH'], 'Plots', 'makeSplusBModelPlot.py')}",
                "--inputWSFile", datacard_path,
                "--cats", f"{','.join(reco_cats_with_bmw)}",
                "--lumiLabel", get_lumi_label(self.year),
                "--isPreliminary",
                "--doZeroes",
                "--translateCats", f"{os.path.join(os.environ['ANALYSIS_PATH'], 'Plots', 'cats.json')}",
                "--doSumCategories",
                # merge per year cat together
                # "--cats Y22_BEST,Y23_BEST,Y24_BEST,Y25_BEST" 
                "--doCatWeights",
                "--saveWeights",
                "--blindingRegion", "115,135", 
                "--mass", f"{HIGGS_MASS}",
                "--ext", f"_{cat}",
                "--POI", f"{cat}"
            ]
            if skipIndivCat:
                arguments.append("--skipIndividualCatPlots")

            if config['combine_mggToys']['doBands']:
                arguments.append("--doBands")
                arguments.append("--doToyVeto")
                arguments.append("--saveToyYields")
            command = arguments
            print(' '.join(command))
            try:
                result = subprocess.run(command, check=True, text=True, capture_output=True)
                print("Script output:", result.stdout)
                print("Script executed successfully.")
            except subprocess.CalledProcessError as e:
                print("Error executing script:", e.stderr)

        # Copy the files back to pnfs if we are on slurm/psi
        if self.batch_flavor == "slurm/psi":
            # Have to copy over the output to the final directory
            # Don't forget to VOMS!
            if "/work" in output_dir:
                slurm_copy_command = [
                    'cp', '-rf',
                    f"{os.environ['TARGET_PATH']}/Combine/",
                    output_dir
                ]
            else:
                slurm_copy_command = [
                    'xrdcp', '-rf',
                    f"{os.environ['TARGET_PATH']}/Combine/",
                    'root://t3dcachedb03.psi.ch:1094//'+output_dir
                ]
            print(slurm_copy_command)
            execute_command(slurm_copy_command)
            # Clean up the temporary directory
            shutil.rmtree(os.environ["TARGET_PATH"])

        os.chdir(main_dir)
        
        
class PValueCalculation(Task, law.LocalWorkflow, HTCondorWorkflow, SlurmWorkflow): #(law.Task): #(Task, HTCondorWorkflow, law.LocalWorkflow):
    output_dir = law.Parameter(default = '', description="Path to the output directory")
    variable = law.Parameter(default="", description="Variable to be used")
    year = law.Parameter(default='2022', description="Year")

    batch_flavor = law.Parameter(default="htcondor", description="Batch system to use")

    # def requires(self):
    def workflow_requires(self):
        workflow_reqs = super().workflow_requires()

        tasks = {}

        if workflow_reqs:
            tasks.update(workflow_reqs)
        
        config = self.get_input_config()
        output_dir = self.get_output_dir()
        
        fitConfig = config["combine_fit"]
        tasks["CreateUnblindedFit"] = CreateUnblindedFit(output_dir=output_dir, variable=self.variable, year=self.year, batch_flavor=self.batch_flavor, version=f"{self.variable if self.variable != '' else 'inclusive'}", workflow=fitConfig["execution"], slurm_partition=fitConfig['batchPartition'], slurm_memory=fitConfig['batchMemory'], slurm_max_runtime=fitConfig['batchMaxRuntime'], htcondor_partition=fitConfig['batchPartition'], htcondor_memory=fitConfig['batchMemory'], htcondor_max_runtime=fitConfig['batchMaxRuntime'])
        
        return tasks

    def create_branch_map(self):
        # if self.variable == '':
        #     cat_list = ["r"]
        # else:
        #     cat_list = combineVariableDict(self.variable, self.year)['paramStrNoOne']
        # branch_map = {i: cat for i, cat in enumerate(cat_list)}
        branch_map = {i: value for i, value in enumerate([0])}
        return branch_map

    def output(self):
        
        config = self.get_input_config()
        output_dir = self.get_output_dir()

        if self.variable == '':
            fitFolderName = f'runFits_mu_fiducial'
            output = []
        else:
            fitFolderName = f'runFits_{self.variable}'
            output = [os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'dataFit', f'higgsCombine.pvalue.MultiDimFit.mH{HIGGS_MASS}.root')]
            output += [os.path.join(output_dir, 'Combine', outdir, fitFolderName, f'pvalue.txt')]

        outputFileTargets = []
                
        for _, current_output_path in enumerate(output):
            outputFileTargets.append(law.LocalFileTarget(current_output_path))
            
        # print(outputFileTargets)

        return outputFileTargets
    
    def run(self):
       
        if self.variable == '':
            fitFolderName = f'runFits_mu_fiducial'

        else:
            fitFolderName = f'runFits_{self.variable}'

        #Load central config file
        with open(configYamlPath, 'r') as file:
            config = yaml.safe_load(file)

        if self.output_dir == '':
            output_dir = config['outputFolder']
        else:
            output_dir = self.output_dir  
            
        cwd = os.getcwd()
        if self.batch_flavor == "slurm/psi":
            # Have to use /scratch/batch_username/ for slurm/psi
            if "/work" in output_dir:
                execute_command([f'mkdir -p {output_dir}/Combine/{fitFolderName}/dataFit'], shell=True)
            else:   
                execute_command([f'xrdfs root://t3dcachedb03.psi.ch:1094/ mkdir -p {output_dir}/Combine/{fitFolderName}/dataFit'], shell=True)

            os.environ["TARGET_PATH"] = f"/scratch/{os.environ['USER']}/{os.environ['SLURM_JOB_ID']}"
            execute_command([f'mkdir -p $TARGET_PATH/Combine/{fitFolderName}/dataFit'], shell=True)
            os.chdir(os.path.join(os.environ["TARGET_PATH"], 'Combine', fitFolderName, 'dataFit'))
            temp_output_dir = os.environ["TARGET_PATH"]
        else:
            execute_command([f'mkdir -p {output_dir}/Combine/{fitFolderName}/dataFit'], shell=True)
            os.chdir(os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'dataFit'))
            temp_output_dir = output_dir
                    
        # Define the file to check
        pvalue_file = os.path.join(temp_output_dir, 'Combine', outdir, fitFolderName, 'dataFit', f'higgsCombine.pvalue.MultiDimFit.mH{HIGGS_MASS}.root')
        # Check if the file exists
        if not os.path.isfile(pvalue_file):
            print("The pvalue file does not exist in the current directory. Creating it...")
            # Iterate over the parameters
            command = [
                "combine",
                "-M", "MultiDimFit",
                os.path.join(output_dir, 'Combine', outdir, fitFolderName, 'dataFit', f"higgsCombineDataPostFitScanFit_{combineVariableDict(self.variable, self.year)['paramStrNoOne'][0]}.MultiDimFit.mH{HIGGS_MASS}.root"),
                "--algo", "fixed",
                "--X-rtd", "MINIMIZER_freezeDisassociatedParams",
                "--X-rtd", "MINIMIZER_multiMin_hideConstants",
                "--X-rtd", "MINIMIZER_multiMin_maskConstraints",
                "--X-rtd", "MINIMIZER_multiMin_maskChannels=2",
                "--cminDefaultMinimizerStrategy=0",
                "--cminFallbackAlgo", "Minuit2,Simplex,0:0.1",
                "--cminFallbackAlgo", "Minuit2,Combined,0:0.1",
                "--freezeParameters", "MH",
                "--fixedPointPOIs", f"{','.join(combineVariableDict(self.variable, self.year)['paramStr'])},MH=125.07",
                "-n", ".pvalue",
                "-m", f"{HIGGS_MASS}",
                "--saveWorkspace"
            ]
            try:
                result = subprocess.run(command, check=True, text=True, capture_output=True)
                print("Script output:", result.stdout)
                print("Script executed successfully.")
            except subprocess.CalledProcessError as e:
                print("Error executing script:", e.stderr)
        else:
            print("The pvalue file exists in the current directory. Skipping the creation.")
            
        # Define variables
        file_path = os.path.realpath(pvalue_file)


        # Count the number of elements in paramStrNoOne
        n_bins = len(combineVariableDict(self.variable, self.year)['paramStrNoOne'])

        # Change directory to the fitFolderName
        os.chdir("../")

        def calculate_pvalue(filename, n_bins):
            """
            Function to calculate the p-value given a ROOT file and number of bins.
            """
            try:
                # Open the ROOT file and read the deltaNLL values
                nll_data = uproot.open(filename)["limit"].arrays()
                nll = nll_data['deltaNLL'][1]  # Extract the second value in deltaNLL array

                # Compute the p-value
                chi2pdf = chi2(n_bins)
                pval = 1 - chi2pdf.cdf(2 * nll)

                return pval
            except Exception as e:
                print(f"Error while calculating p-value: {e}")
                return None
        # Calculate the p-value
        pvalue = calculate_pvalue(file_path, n_bins)
        
        # Rounding to two significant digits
        pvalue = round(pvalue, 2 - int(f"{pvalue:.1e}".split('e')[1]) - 1)

        if pvalue is not None:
            # Define your differential variable for printing
            output_file = "pvalue.txt"
            try:
                with open(output_file, "w") as f:
                    f.write(f"{pvalue}")
                print(f"P-value written to {output_file}")
            except Exception as e:
                print(f"Error writing to file: {e}")
            print(f"The p-value of the variable {self.variable} is: {pvalue}")
        else:
            print("Failed to calculate the p-value.")

        # Copy the files back to pnfs if we are on slurm/psi
        if self.batch_flavor == "slurm/psi":
            # Have to copy over the output to the final directory
            # Don't forget to VOMS!
            if "/work" in output_dir:
                slurm_copy_command = [
                    'cp', '-rf',
                    f"{os.environ['TARGET_PATH']}/Combine/",
                    output_dir
                ]
            else:
                slurm_copy_command = [
                    'xrdcp', '-rf',
                    f"{os.environ['TARGET_PATH']}/Combine/",
                    'root://t3dcachedb03.psi.ch:1094//'+output_dir
                ]
            print(slurm_copy_command)
            execute_command(slurm_copy_command)
            # Clean up the temporary directory
            shutil.rmtree(os.environ["TARGET_PATH"])

        os.chdir(cwd)
