import law
import luigi
import os
import re
import subprocess
import ROOT
import uproot
import glob
import yaml
import json
import errno
import shutil
import warnings
import math
# Suppress // UserWarning: The value of the smallest subnormal for <class 'numpy.float64'> type is zero // warning.
warnings.filterwarnings("ignore", category=UserWarning, module="numpy.core.getlimits")
from scipy.stats import chi2

from commonTools import *
from commonObjects import *
# Helpers to extract/save pdfindex information for merged category flows
from pdfindex_utils import extract_pdf_indices, update_override_file

from Datacard.law_datacard import *
from Background.law_background import *

from framework import Task, MultiYearTask
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

HIGGS_MASS = 125.07

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

class CombineTask(Task):

    @property
    def fitFolderName(self):
        return f"runFits_{self.variable if self.variable else 'mu_fiducial'}"

    @property
    def datacard_path(self):
        output_dir = self.get_output_dir()
        if self.variable == '':
            datacard_path = os.path.join(output_dir, 'Combine', self.outdir, f'Datacard_{self.year}.root')
        else:
            datacard_path = os.path.join(output_dir, 'Combine', self.outdir, f'Datacard_{self.variable}_{self.year}.root')
        return datacard_path

class CombineMultiYearTask(MultiYearTask):

    @property
    def fitFolderName(self):
        return f"runFits_{self.variable if self.variable else 'mu_fiducial'}"

    @property
    def datacard_path(self):
        output_dir = self.get_output_dir()
        if self.variable == '':
            datacard_path = os.path.join(output_dir, 'Combine', self.outdir, f'Datacard_{self.year}.root')
        else:
            datacard_path = os.path.join(output_dir, 'Combine', self.outdir, f'Datacard_{self.variable}_{self.year}.root')
        return datacard_path



    
class PrepareTheDirectory(CombineMultiYearTask):#(law.Task): #(CombineTask, HTCondorWorkflow, law.LocalWorkflow):

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
                
        # Load the input configuration
        config = self.get_input_config()
        output_dir = self.get_output_dir()


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

        output_data.append(os.path.join(output_dir, 'Combine', self.outdir))

        output_data.append(os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName))

        years = yearMap[self.year]
        if len(years) > 1:
            raise RuntimeError(f"PrepareTheDirectory expects a single year, but got: {years}. Run CombineDatacards instead.")


        if signal_model_folder_name == background_model_folder_name:
            model_folder_name = signal_model_folder_name
            output_data.append(os.path.join(output_dir, 'Combine', self.outdir, model_folder_name))
            output_data.append(os.path.join(output_dir, 'Combine', self.outdir, model_folder_name, 'background'))
            output_data.append(os.path.join(output_dir, 'Combine', self.outdir, model_folder_name, 'signal'))
        else:
            output_data.append(os.path.join(output_dir, 'Combine', self.outdir, signal_model_folder_name))
            output_data.append(os.path.join(output_dir, 'Combine', self.outdir, signal_model_folder_name, 'signal'))

            output_data.append(os.path.join(output_dir, 'Combine', self.outdir, background_model_folder_name))
            output_data.append(os.path.join(output_dir, 'Combine', self.outdir, background_model_folder_name, 'background'))

        for cat in cat_list:
            output_data.append(os.path.join(output_dir, 'Combine', self.outdir, background_model_folder_name, 'background', f'CMS-HGG_multipdf_{cat}.root'))
            output_data.append(os.path.join(output_dir, 'Combine', self.outdir, signal_model_folder_name, 'signal', f'CMS-HGG_sigfit_packaged{outputExt}_{cat}.root'))


        # Define the file paths
        if self.variable == '':
            output_data.append(os.path.join(output_dir, 'Combine', self.outdir, f'Datacard_{self.year}.txt'))
        else:
            output_data.append(os.path.join(output_dir, 'Combine', self.outdir, f'Datacard_{self.variable}_{self.year}.txt'))

        for i, output in enumerate(output_data):
            output_data[i] = law.LocalFileTarget(output)

        return output_data

    def run(self):
        background_suffix = f""


        #Load central config file
        config = self.get_input_config()
        output_dir = self.get_output_dir()

        # Creating the Combine directory alongside the Models dir
        safe_mkdir(os.path.join(output_dir, 'Combine', self.outdir))
        safe_mkdir(os.path.join(output_dir, 'Combine', self.outdir))
        safe_mkdir(os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName))

        signal_model_folder_name = config['datacard_yields']['sigModelWSDir'].split('/')[-2]
        background_model_folder_name = config['datacard_yields']['bkgModelWSDir'].split('/')[-2]

        years = yearMap[self.year]
        if len(years) > 1:
            raise RuntimeError(f"PrepareTheDirectory expects a single year, but got: {years}. Run CombineDatacards instead.")
        
    
        if signal_model_folder_name == background_model_folder_name:
            Model_dst_path = os.path.join(output_dir, 'Combine', self.outdir, signal_model_folder_name)
            background_dst_path = os.path.join(output_dir, 'Combine', self.outdir, signal_model_folder_name, 'background'+background_suffix)
            signal_dst_path = os.path.join(output_dir, 'Combine', self.outdir,signal_model_folder_name, 'signal')
            safe_mkdir(Model_dst_path)
        else:
            signalModel_dst_path = os.path.join(output_dir, 'Combine', self.outdir, signal_model_folder_name)
            signal_dst_path = os.path.join(output_dir, 'Combine', self.outdir, signal_model_folder_name, 'signal')
            safe_mkdir(signalModel_dst_path)
            
            backgroundModel_dst_path = os.path.join(output_dir, 'Combine', self.outdir, background_model_folder_name)
            background_dst_path = os.path.join(output_dir, 'Combine', self.outdir, background_model_folder_name, 'background'+background_suffix)
            safe_mkdir(backgroundModel_dst_path)

        safe_mkdir(signal_dst_path)
        safe_mkdir(background_dst_path)

        # Copying relevant files in Models directory
        background_src_path = os.path.join(output_dir, "Background", f"outdir_{config['backgroundScriptCfg']['ext']}"+background_suffix)
        signal_src_path = os.path.join(output_dir, 'Signal', f"outdir_packaged{config[f'packaged_{self.year}']['ext']}_{self.year}/")

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
            destination_file = os.path.join(output_dir, 'Combine', self.outdir, f'Datacard_{self.year}.txt')
        else:
            datacard_file_cleaned = os.path.join(output_dir, 'Datacards', f'Datacard_{self.variable}_{self.year}_cleaned.txt')
            datacard_file = os.path.join(output_dir, 'Datacards', f'Datacard_{self.variable}_{self.year}.txt')
            destination_file = os.path.join(output_dir, 'Combine', self.outdir, f'Datacard_{self.variable}_{self.year}.txt')

        # Check if the cleaned file exists
        if os.path.exists(datacard_file_cleaned):
            # Copy the cleaned file if it exists
            shutil.copy2(datacard_file_cleaned, destination_file)
        else:
            # Otherwise, copy the uncleaned file
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

        combined_label = self.year
        
        years = yearMap[self.year]

        combined_output_dir = Path(output_dir) / "Combine" / self.outdir

        
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

        if self.variable == '':
            combined_card_path = os.path.join(
                base_dir,
                f"output_{combined_label}_inclusive/Combine/{self.outdir}/Datacard_{combined_label}.txt"
            )
        else:
            combined_card_path = os.path.join(
                base_dir,
                f"output_{combined_label}_{self.variable}/v{self.version}/Combine/{self.outdir}/Datacard_{self.variable}_{combined_label}.txt"
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

        # Define the file paths
        if self.variable == '':
            output = os.path.join(output_dir, 'Combine', self.outdir, f'Datacard_{self.year}.root')
        else:
            output = os.path.join(output_dir, 'Combine', self.outdir, f'Datacard_{self.variable}_{self.year}.root')

        return law.LocalFileTarget(output)
        
       

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
        
        temp_output_dir = output_dir
        datacards_dir = os.path.join(temp_output_dir, 'Combine', self.outdir)

        arguments = [
            "python3",
            script_path,
            "--inputName", datacard_name,
            "--outputDir", datacards_dir,
            "--outputName", workspace_name,
            "--mode", mode,
            "--common_opts", f"-m {HIGGS_MASS} higgsMassRange=122,128 --channel-masks", # higgsMassRange is redefined in model.py:floatingHiggsMass
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

        # Persist the pdfindex values from the produced workspace so downstream
        # cat-merged fits can reuse the same indices without recomputing them.
        final_root_path = os.path.join(output_dir, 'Combine', self.outdir, f'{datacard_name}.root')
        pdf_indices = extract_pdf_indices(final_root_path)
        if pdf_indices:
            override_path = os.path.join(os.environ["ANALYSIS_PATH"], "config", "pdfindex_overrides.json")
            variable_key = self.variable if self.variable != '' else 'inclusive'
            update_override_file(override_path, self.year, variable_key, pdf_indices)
        
        
class AsimovFitCategoryFirstStep(CombineTask, law.LocalWorkflow, HTCondorWorkflow, SlurmWorkflow): #(law.Task): #(CombineTask, HTCondorWorkflow, law.LocalWorkflow):
    cats = law.Parameter(description="Current category")
    
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

            
        output = {}
        output['scan_folder'] = law.LocalFileTarget(os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'asimov'))

        output['first_step_root'] = law.LocalFileTarget(os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'asimov', f'higgsCombinefirstStep_{current_branch}.MultiDimFit.mH{HIGGS_MASS}.root'))
        output['first_step_multidim'] = law.LocalFileTarget(os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'asimov', f'multidimfitfirstStep_{current_branch}.root'))
        

        return output

    def run(self):
        current_branch = self.branch_data

        output_dir = self.get_output_dir()
  
            

        cwd = os.getcwd()

        fit_dir = os.path.join(
            output_dir, "Combine", self.outdir, self.fitFolderName, "asimov"
        )
        os.makedirs(fit_dir, exist_ok=True)
        os.chdir(fit_dir)

        # Split the year string into a list
        years = self.year.split("_")

        if self.variable in ["", "tuto"]:
            # Make all combinations of BMW and years: This also works if self.year is 2022_2023 in a combineCards workflow!
            pdf_indices = [f"pdfindex_{bmw}_{year}_13TeV" for bmw in BMW for year in years]

            arguments = [
                "combine",
                "-M", "MultiDimFit",
                self.datacard_path,
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
                cache_key = (self.datacard_path,)
                if cache_key not in _PDFINDEX_CACHE and os.path.exists(self.datacard_path):
                    datacard_pdf_indices = extract_pdf_indices(self.datacard_path)
                    if datacard_pdf_indices:
                        _PDFINDEX_CACHE[cache_key] = datacard_pdf_indices
                if cache_key in _PDFINDEX_CACHE:
                    pdf_indices = _PDFINDEX_CACHE[cache_key]
            else:
                pdf_indices = None
                
            arguments = [
                "combine",
                "-M", "MultiDimFit",
                self.datacard_path,
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
        
        os.chdir(cwd)


class FitCategoryFirstStep(CombineTask):
    """Common first-step adapter for expected and observed scans."""

    freeze = law.Parameter(default="")
    is_postfit = luigi.BoolParameter(default=False)

    def requires(self):
        output_dir = self.get_output_dir()
        if not self.is_postfit:
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
            
            return AsimovFitCategoryFirstStep.req(
                self, 
                output_dir=output_dir, 
                cats=cats, 
                # workflow=impactConfig["execution"], 
                # slurm_partition=impactConfig['batchPartition'], 
                # slurm_memory=impactConfig['batchMemory'], 
                # slurm_max_runtime=impactConfig['batchMaxRuntime'], 
                # htcondor_partition=impactConfig['batchPartition'], 
                # htcondor_memory=impactConfig['batchMemory'], 
                # htcondor_max_runtime=impactConfig['batchMaxRuntime']
                )
                
        else:

            fit_config = self.get_input_config()["combine_fit"]
            return UnblindedFitFirstStep.req(
                self,
                output_dir=output_dir,
                # workflow=fit_config["execution"],
                # slurm_partition=fit_config["batchPartition"],
                # slurm_memory=fit_config["batchMemory"],
                # slurm_max_runtime=fit_config["batchMaxRuntime"],
                # htcondor_partition=fit_config["batchPartition"],
                # htcondor_memory=fit_config["batchMemory"],
                # htcondor_max_runtime=fit_config["batchMaxRuntime"],
            )

    def output(self):
        return self.input()

    def run(self):
        return True

# Handles both standard per-category scans and the merged-category flow by
# optionally branching over a comma-separated list of categories. When
# `cats` is set all bins are executed inside one HTCondor job to avoid
# spawning one submission per differential bin.
class FitCategoryScan(CombineTask, law.LocalWorkflow, HTCondorWorkflow, SlurmWorkflow):
    cat = law.Parameter(default="", description="Current category")
    cats = law.Parameter(default="", description="Comma separated list of categories to process")
    nPoints = law.Parameter(default=30, description="Number of points for the LL scan")
    set_pdfidx_inclusives = law.Parameter(default=False, description="Year") # convert_boolean_string
    freeze = law.Parameter(default="", description="Parameters to freeze in the fit")
    is_postfit = luigi.BoolParameter(
        default=False,
        description="Use observed data and post-fit snapshots instead of Asimov data",
    )

    def requires(self):
        return self.workflow_requires()
    
    def workflow_requires(self):
        workflow_reqs = super().workflow_requires()

        tasks = {}

        if workflow_reqs:
            tasks.update(workflow_reqs)
        
        output_dir = self.get_output_dir()
               
        tasks["FitCategoryFirstStep"] = FitCategoryFirstStep.req(
            self,
            output_dir=output_dir,
            freeze=self.freeze,
            is_postfit=self.is_postfit,
        )
        
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


            
        mode_dir = 'dataFit' if self.is_postfit else 'asimov'
        scan_name = 'DataPostFitScanFit' if self.is_postfit else 'AsimovPostFitScanFit'
        output = [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, mode_dir, current_dir, self.freeze if self.freeze else 'NoFreeze')]
        output += [os.path.join(
            output_dir, 'Combine', self.outdir, self.fitFolderName, mode_dir, current_dir, self.freeze if self.freeze else 'NoFreeze',
            f'higgsCombine{scan_name}_{self.cat}.POINTS.{current_point}.{current_point}.MultiDimFit.mH{HIGGS_MASS}.root',
        )]
        
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
    
    def store_parts(self):
        scan_id = law.util.create_hash(
            (
                self.year,
                self.variable,
                self.cat,
                self.cats,
                self.freeze,
                self.is_postfit,
                self.set_pdfidx_inclusives,
            )
        )

        return (
            self.__class__.__name__,
            self.version,
            scan_id,
        )

    def run(self):
        current_cat, current_point = self._current_branch_info()
        current_dir = ("_".join(current_cat.split(",")) if current_cat != self.cat else "")

        config = self.get_input_config()

        scan_prefix = ("DataPostFitScanFit" if self.is_postfit else "AsimovPostFitScanFit")

        scan_dir = os.path.join(self.input()['FitCategoryFirstStep']['collection'][0]['scan_folder'].path, current_dir, self.freeze if self.freeze else 'NoFreeze')
        os.makedirs(scan_dir, exist_ok=True)

        first_step = self.input()['FitCategoryFirstStep']['collection'][0]['first_step_root'].path

        if not os.path.isfile(first_step):
            raise RuntimeError(f"Required first-step workspace is missing: {first_step}")

        frozen = ["r" if self.variable == "MH" else "MH"]

        if self.freeze == "allConstrainedNuisances":
            frozen.append("allConstrainedNuisances")

        arguments = [
            "combineTool.py",
            "-M", "MultiDimFit",
            "-d", first_step,
            "-m", str(HIGGS_MASS),
            "-n", f"{scan_prefix}_{self.cat}.POINTS.{current_point}.{current_point}",
            "--algo", "grid",
            "--points", str(int(self.nPoints)),
            "--firstPoint", str(current_point),
            "--lastPoint", str(current_point),
            "-P", self.cat,
            "--floatOtherPOIs", "1",
            "--alignEdges", "1",
            "--saveWorkspace",
            "--saveFitResult",
            "--freezeParameters", ",".join(frozen),
            "--cminDefaultMinimizerStrategy=0",
            "--cminFallbackAlgo", "Minuit2,Migrad,1:10",
            "--X-rtd", "MINIMIZER_freezeDisassociatedParams",
            "--X-rtd", "MINIMIZER_multiMin_hideConstants",
            "--X-rtd", "MINIMIZER_multiMin_maskConstraints",
            "--X-rtd", "MINIMIZER_multiMin_maskChannels=2",
        ]

        # Expected/Asimov dataset.
        if not self.is_postfit:
            arguments += [
                "-t", "-1",
                "--expectSignal", "1",
            ]

        # Load the saved fit snapshot.
        if self.is_postfit or self.variable != "":
            arguments += ["--snapshotName", "MultiDimFit"]

        if self.variable == "MH":
            arguments += [
                "--redefineSignalPOIs", "MH",
                "--setParameters", "r=1",
            ]

            if self.cats:
                arguments[-1] += "," + self.make_channel_mask_params(self.cats)

        elif self.variable in ("", "tuto"):
            arguments += ["--setParameters", "r=1"]

        else:
            set_parameters = ",".join(combineVariableDict(self.variable, self.year)["paramStr"])
            arguments += ["--setParameters", set_parameters]

            pdf_indices = combineVariableDict(self.variable, self.year)["pdfIndeces"]
            arguments.extend(_save_specified_index_args(pdf_indices))

        if self.freeze and self.freeze != "allConstrainedNuisances":
            arguments += ["--freezeNuisanceGroups", ",".join(self.freeze.split("_"))]

        arguments.extend(_scan_parameter_range_args(config, current_cat))

        print(" ".join(arguments))
        subprocess.run(arguments, cwd=scan_dir, check=True, text=True)
        
        
class CreateFit(CombineMultiYearTask, law.LocalWorkflow, HTCondorWorkflow, SlurmWorkflow):

    group = law.Parameter(default="Syst,Stat")
    set_pdfidx_inclusives = law.Parameter(default=False)
    cats = law.Parameter(default="")
    is_postfit = luigi.BoolParameter(default=False)


    def _pois(self):
        if self.variable in ("", "tuto"):
            return ["r"]
        if self.variable == "MH":
            return ["MH"]
        return combineVariableDict(self.variable, self.year)["paramStrNoOne"]

    def _groups(self):
        return [group.strip() for group in self.group.split(",") if group.strip()]

    def _freeze_scenarios(self):
        groups = self._groups()
        return [
            "allConstrainedNuisances"
            if index == len(groups) - 1
            else "_".join(groups[:index])
            for index in range(len(groups))
        ]

    def _n_points(self, config):
        key = "unblindedFit_numPoints" if self.is_postfit else "asimov_numPoints"
        return int(config["combine_fit"][key])

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
            "nPoints": config_combine[
                "unblindedFit_numPoints" if self.is_postfit else "asimov_numPoints"
            ],
            "workflow": config_combine["execution"],
            "slurm_partition": config_combine["batchPartition"],
            "slurm_memory": config_combine["batchMemory"],
            "slurm_max_runtime": config_combine["batchMaxRuntime"],
            "htcondor_partition": config_combine["batchPartition"],
            "htcondor_memory": config_combine["batchMemory"],
            "htcondor_max_runtime": config_combine["batchMaxRuntime"],
        }

        if self.variable in ("", "tuto"):
            pois = ["r"]
        elif self.variable == "MH":
            pois = ["MH"]
        else:
            pois = combineVariableDict(self.variable, self.year)["paramStrNoOne"]

        for i, syst in enumerate(self.group.split(',')):
            if i == len(self.group.split(','))-1:
                freeze = "allConstrainedNuisances"
            else:
                freeze = '_'.join(self.group.split(",")[:i])

            for poi in pois:
                tasks[f"FitCategory_{syst}_{poi}"] = FitCategoryScan.req(
                    self,
                    output_dir=output_dir,
                    cat=poi,
                    freeze=freeze,
                    is_postfit=self.is_postfit,
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


    def output(self):
        output_dir = self.get_output_dir()
        current_dir = "_".join(self.cats.split(","))



        mode_dir = "dataFit" if self.is_postfit else "asimov"
        base_dir = os.path.join(output_dir, "Combine", self.outdir, self.fitFolderName, mode_dir)

        scan_dir = os.path.join(base_dir, "scans", current_dir,
        )

        if self.variable in ["", "tuto", "MH"]:
            pois = ["MH"] if self.variable == "MH" else ["r"]
        else:
            pois = combineVariableDict(self.variable, self.year)["paramStrNoOne"]

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
                path = os.path.join(base_dir, current_dir, freeze if freeze else 'NoFreeze', f"higgsCombine{'Data' if self.is_postfit else 'Asimov'}PostFitScanFit_{poi}.root")
                outputs["scans"][scan_type][poi] = law.LocalFileTarget(path)

        # plot outputs are produced only once
        group_name = "_".join(groups)

        for poi in pois:
            outputs["plots"][poi] = {}
            for ext in ("root", "pdf", "png"):
                path = os.path.join(scan_dir, f"scan_{poi}_{group_name}.{ext}")
                outputs["plots"][poi][ext] = law.LocalFileTarget(path)

        return outputs

    def _run_observed(self):
        """Aggregate and plot observed scan points in the common layout."""
        config = self.get_input_config()
        output_dir = self.get_output_dir()
        fit_folder = 'runFits_mu_fiducial' if self.variable == '' else f'runFits_{self.variable}'
        
        current_dir = '_'.join(self.cats.split(','))
        base_dir = os.path.join(output_dir, 'Combine', self.outdir, fit_folder, 'dataFit')
        scan_dir = os.path.join(base_dir, 'scans', current_dir)
        os.makedirs(scan_dir, exist_ok=True)

        if self.variable in ('', 'tuto'):
            pois = ['r']
        elif self.variable == 'MH':
            pois = ['MH']
        else:
            pois = combineVariableDict(self.variable, self.year)['paramStrNoOne']

        groups = self.group.split(',')
        n_points = int(config['combine_fit']['unblindedFit_numPoints'])
        for poi in pois:
            freeze_list = []
            for index, _ in enumerate(groups):
                freeze = (
                    'allConstrainedNuisances'
                    if index == len(groups) - 1
                    else '_'.join(groups[:index])
                )
                freeze_list.append(freeze)
                point_dir = os.path.join(base_dir, current_dir, freeze)
                merged = os.path.join(
                    point_dir, f'higgsCombineDataPostFitScanFit_{poi}.root'
                )
                sources = [
                    os.path.join(
                        point_dir,
                        f'higgsCombineDataPostFitScanFit_{poi}.POINTS.{point}.{point}.MultiDimFit.mH{HIGGS_MASS}.root',
                    )
                    for point in range(n_points)
                ]
                subprocess.run(
                    ['hadd', '-f', merged] + sources,
                    check=True,
                    text=True,
                    capture_output=True,
                )

            total_scan = os.path.join(
                base_dir, current_dir, freeze_list[0] if freeze_list[0] else 'NoFreeze',
                f'higgsCombineDataPostFitScanFit_{poi}.root',
            )
            arguments = [
                'python3',
                os.path.join(os.environ['ANALYSIS_PATH'], 'Plots', 'plot1DScan.py'),
                total_scan,
                '-o', f'scan_{poi}_{"_".join(groups)}',
                '--POI', poi,
                '--main-label', 'Observed',
                '--translate', os.path.join(
                    os.environ['ANALYSIS_PATH'], 'Combine', 'pois.json'
                ),
                '--breakdown', self.group,
                '--x-min', str(config['combine_fit'].get('xMin', 124)),
                '--x-max', str(config['combine_fit'].get('xMax', 126)),
                '--others',
            ]
            for index, freeze in enumerate(freeze_list[1:]):
                label = (
                    'Stat only' if freeze == 'allConstrainedNuisances'
                    else 'freeze ' + '+'.join(freeze.split('_'))
                )
                other_scan = os.path.join(
                    base_dir, current_dir, freeze,
                    f'higgsCombineDataPostFitScanFit_{poi}.root',
                )
                arguments.append(f'{other_scan}:{label}:{index + 2}')
            print(' '.join(arguments))
            subprocess.run(arguments, cwd=scan_dir, check=True, text=True)

    def run(self):
        if self.is_postfit:
            return self._run_observed()

        config = self.get_input_config()
        output_dir = self.get_output_dir()
        current_dir = "_".join(self.cats.split(","))

        cwd = os.getcwd()
        scan_dir = os.path.join(
            output_dir, "Combine", self.outdir, self.fitFolderName,
            "asimov", "scans", current_dir,
        )
        os.makedirs(scan_dir, exist_ok=True)
        os.chdir(os.path.join(
            output_dir, "Combine", self.outdir, self.fitFolderName,
            "asimov", current_dir,
        ))

        for cat in self._pois():
            freeze_list = self._freeze_scenarios()
            # hadd the files for each category and freeze combination
            for freeze in freeze_list:
                freeze = freeze if freeze else 'NoFreeze'
                arguments = [
                    "hadd", "-f",
                    f"{os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'asimov', current_dir, freeze, f'higgsCombineAsimovPostFitScanFit_{cat}.root')}"
                ]
                for i in range(self._n_points(config)):
                    arguments.append(os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'asimov', current_dir, freeze, f'higgsCombineAsimovPostFitScanFit_{cat}.POINTS.{i}.{i}.MultiDimFit.mH{HIGGS_MASS}.root'))
                command = arguments
                # print(command)
                try:
                    result = subprocess.run(command, check=True, text=True, capture_output=True)
                    print("Script output:", result.stdout)
                    print("Script executed successfully.")
                except subprocess.CalledProcessError as e:
                    print("Error executing script:", e.stderr)
                    raise
                    
                

            os.chdir(scan_dir)

            arguments = [
                "python3", 
                # f"{os.environ['CMSSW_BASE']}/bin/{os.environ['SCRAM_ARCH']}/plot1DScan.py",
                os.path.join(os.environ["ANALYSIS_PATH"],"Plots/plot1DScan.py"),
                # "plot1DScan.py",
                os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'asimov', current_dir, freeze_list[0] if freeze_list[0] else 'NoFreeze', f'higgsCombineAsimovPostFitScanFit_{cat}.root'),
                "-o", f'scan_{cat}_{"_".join(self.group.split(","))}',
                "--POI", f"{cat}",
                "--main-label", "Expected",
                "--translate", os.path.join(os.environ["ANALYSIS_PATH"], 'Combine', 'pois.json'),
                "--breakdown", self.group,
                "--x-min", str(config["combine_fit"].get("xMin", 124)),
                "--x-max", str(config["combine_fit"].get("xMax", 126)),
                "--others", 
            ]

            arguments += [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'asimov', current_dir, freeze, f'higgsCombineAsimovPostFitScanFit_{cat}.root')+f":{'Stat only' if freeze=='allConstrainedNuisances' else 'freeze '+'+'.join(freeze.split('_'))}:{i+2}" for i, (freeze, name) in enumerate(zip(freeze_list[1:], self.group.split(',')[1:]))]

            command = arguments
            print(' '.join(command))
            try:
                result = subprocess.run(command, check=True, text=True, capture_output=True)
                print("Script output:", result.stdout)
                print("Script executed successfully.")
            except subprocess.CalledProcessError as e:
                print("Error executing script:", e.stderr)
    
        os.chdir(cwd)


class CreateLikelihoodFitWrapper(CombineMultiYearTask):

    set_pdfidx_inclusives = law.Parameter(default=False)
    do_per_cat = luigi.BoolParameter(default=False, description="Run per-category fits instead of inclusive fits")
    do_splitting = luigi.BoolParameter(default=False, description="Do splitting the group list for per-category fits, only use the first group")
    is_postfit = luigi.BoolParameter(default=False)

    
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
        }

        # Inclusive fits: preserve all configured groups
        for group in groups:
            label = group.replace(",", "_")

            tasks["inclusive"][label] = CreateFit.req(
                self,
                years=self.years,
                output_dir=output_dir,
                group=group,
                set_pdfidx_inclusives=self.set_pdfidx_inclusives,
                workflow=self.batch_flavor,
                cats="",
                is_postfit=self.is_postfit,
            )

        if self.do_per_cat:
            # Per-category fits only need the first group,
            # matching your previous behavior.
            if not self.do_splitting:
                groups = groups[0:1]

            for group in groups:
                label = group.replace(",", "_")
                for cat in self.get_cats().split(","):
                    cat = cat.strip()
                    tasks.setdefault(cat, {})

                    if not cat:
                        continue

                    tasks[cat][label] = CreateFit.req(
                        self,
                        years=self.years,
                        output_dir=output_dir,
                        group=group,
                        set_pdfidx_inclusives=self.set_pdfidx_inclusives,
                        workflow=self.batch_flavor,
                        cats=cat,
                        is_postfit=self.is_postfit,
                    )

        return tasks

    def output(self):
        return self.input()

    def run(self):
        return True


class CreateFitPerCat(CombineMultiYearTask):
    include_stat_error = luigi.BoolParameter(default=True, description="Include stat-only scan inputs and print stat/syst breakdown")
    show_values = luigi.BoolParameter(default=True, description="Draw numerical fit values on each summary row")
    is_per_year = luigi.BoolParameter(default=False, description="Show per year breakdown in the summary plot")
    set_pdfidx_inclusives = law.Parameter(default=False)
    is_postfit = luigi.BoolParameter(default=False)

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
                tasks[year] = CreateLikelihoodFitWrapper.req(
                    self,
                    years=year,
                    output_dir=self.get_output_dir(),
                    set_pdfidx_inclusives=self.set_pdfidx_inclusives,
                    do_per_cat=False,
                    is_postfit=self.is_postfit,
                )
            return tasks
        return {self.year: CreateLikelihoodFitWrapper.req(
            self,
            years=self.years,
            output_dir=self.get_output_dir(),
            set_pdfidx_inclusives=self.set_pdfidx_inclusives,
            do_per_cat=True,
            is_postfit=self.is_postfit,
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
            "dataFit" if self.is_postfit else "asimov",
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
                "CreateFitPerCat is currently intended for variable=MH scans"
            )

        cwd = os.getcwd()

        try:
            # Create output directory and move there
            execute_command([f"mkdir -p {self._summary_dir()}"], shell=True)
            os.chdir(self._summary_dir())

            # Inputs from CreateLikelihoodFitWrapper
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
            groups = config["combine_fit"].get("group", ["Syst,Stat"],)

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
                "python3", os.path.join(os.environ["ANALYSIS_PATH"], "Plots", "plotAsimovScanSummary.py",),
                "--POI", "MH",
                "--output", self._output_base(),
                "--band-from", str(band_from),
                "--split-after", str(split_after),
                "--x-title", "m_{H} (GeV)",
                "--x-min", str(config["combine_fit"].get("xMin", 124,)),
                "--x-max", str(config["combine_fit"].get("xMax", 126,)),
                "--cms-label", "Internal",
                "--lumi-label", get_lumi_label(self.year),
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
                        self._inclusive_scan_argument(inputs[year], main_group_label, label),
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
                    "--scan", self._inclusive_scan_argument(inputs, main_group_label, f"CMS H#gamma#gamma {self.year}"),
                ]

            # ------------------------------------------------------------
            # Run plotting script
            # ------------------------------------------------------------
            print("Running command:")
            print(" ".join(arguments))

            result = subprocess.run(arguments, check=True, text=True, capture_output=True)

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


class AsimovImpactFirstStep(CombineTask, law.LocalWorkflow, HTCondorWorkflow, SlurmWorkflow): #(law.Task): #(CombineTask, HTCondorWorkflow, law.LocalWorkflow):

    # def requires(self):
    def workflow_requires(self):
        workflow_reqs = super().workflow_requires()

        tasks = {}

        if workflow_reqs:
            tasks.update(workflow_reqs)
        
        config = self.get_input_config()
        output_dir = self.get_output_dir()
        
        impactConfig = config["combine_impacts"]    
        
        tasks["RunT2WS"] = RunText2Workspace.req(self, output_dir=output_dir, years=self.year)
        tasks["CreateAsimovFitFirstStep"] = FitCategoryFirstStep.req(self, output_dir=output_dir, )
        
        return tasks

    def create_branch_map(self):
        branch_map = {i: current_branch for i, current_branch in enumerate([0])}
        return branch_map

    def output(self):        
        config = self.get_input_config()
        output_dir = self.get_output_dir()


            
        # output = [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName)]
        output = [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'impact')]

        output += [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'impact', f'higgsCombine_initialFit_Test.MultiDimFit.mH{HIGGS_MASS}.root')]
        
        outputFileTargets = []
                
        for _, current_output_path in enumerate(output):
            outputFileTargets.append(law.LocalFileTarget(current_output_path))
            
        # print(outputFileTargets)

        return outputFileTargets

    def run(self):
       
        config = self.get_input_config()
        output_dir = self.get_output_dir()
        
        cwd = os.getcwd()
        execute_command([f'mkdir -p {output_dir}/Combine/{self.outdir}/{self.fitFolderName}/impact'], shell=True)
        os.chdir(os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'impact'))

        first_output = os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'asimov')
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
        if self.variable != "MH" and os.path.exists(first_step_path):
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
                "-d", self.datacard_path,
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
                "-d", self.datacard_path,
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
                "-d", self.datacard_path,
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
    
        os.chdir(cwd)
        
class AsimovImpactSecondStep(CombineTask, law.LocalWorkflow, HTCondorWorkflow, SlurmWorkflow): #(law.Task): #(CombineTask, HTCondorWorkflow, law.LocalWorkflow):

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

        if os.path.exists(self.datacard_path):
            paramList = list_nuisance_parameters(self.datacard_path, "w", "ModelConfig", poiList, exclude_expr=exclude_expr)
        else:
            paramList = ["dummy_param"]
        
        branch_map = {i: current_param for i, current_param in enumerate(paramList)}
        return branch_map

    def output(self):        
        current_param = self.branch_data
        
        config = self.get_input_config()
        output_dir = self.get_output_dir()

            
        # output = [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName)]
        output = [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'impact')]

        output += [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'impact', f'higgsCombine_paramFit_Test_{current_param}.MultiDimFit.mH{HIGGS_MASS}.root')]
        
        outputFileTargets = []
                
        for _, current_output_path in enumerate(output):
            outputFileTargets.append(law.LocalFileTarget(current_output_path))
            
        # print(outputFileTargets)

        return outputFileTargets

    def run(self):
        current_param = self.branch_data
       

                  
        config = self.get_input_config()
        output_dir = self.get_output_dir()             
        
        cwd = os.getcwd()
        execute_command([f'mkdir -p {output_dir}/Combine/{self.outdir}/{self.fitFolderName}/impact'], shell=True)
        os.chdir(os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'impact'))

        first_output = os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'asimov')
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
                "-d", self.datacard_path,
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
                "-d", self.datacard_path,
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
                "-d", self.datacard_path,
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
        
        os.chdir(cwd)
        
class AsimovImpactThirdStep(CombineTask, law.LocalWorkflow, HTCondorWorkflow, SlurmWorkflow): #(law.Task): #(CombineTask, HTCondorWorkflow, law.LocalWorkflow):
    
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



        output = []
        if self.variable in ['', 'tuto', 'MH']:
            cat = "r"
            output += [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'impact', 'impacts')]
            output += [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'impact', 'impacts', f'impacts.pdf')]
            if self.variable != 'MH':
                output += [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'impact', 'impacts', 'impacts_corrected_dropBkgModelParams.json')]
            
        else:
            for cat in combineVariableDict(self.variable, self.year)['paramStrNoOne']:
                output += [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'impact', 'impacts')]
                output += [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'impact', 'impacts', f'impacts_{cat}.pdf')]
                
        output += [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'impact', 'impacts', f'impacts.json')]
        
        outputFileTargets = []
                
        for _, current_output_path in enumerate(output):
            outputFileTargets.append(law.LocalFileTarget(current_output_path))
            
        # print(outputFileTargets)

        return outputFileTargets

    def run(self):

                  
        config = self.get_input_config()
        output_dir = self.get_output_dir() 
            


        cwd = os.getcwd()         
        execute_command([f'mkdir -p {output_dir}/Combine/{self.outdir}/{self.fitFolderName}/impact/impacts'], shell=True)
        os.chdir(os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'impact'))
        temp_output_dir = output_dir

        first_output = os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'asimov')
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
        if self.variable != "MH" and os.path.exists(first_step_path):
            pdfIdx = check_pdf_idx(pdf_idx_param)
        else:
            print(f"First-step file not found for PDF index extraction: {first_step_path}")

        set_param_string = None
        if pdfIdx:
            set_param_string = f"{base_param_string},{pdfIdx}"

        if self.variable in ['', 'tuto']:
            exclude_expr = config.get("combine_impacts", {}).get("exclude", "")
            named_params = _list_modelconfig_nuisances(self.datacard_path, ["r"], exclude_expr=exclude_expr)
            if not named_params:
                raise RuntimeError(f"No nuisance parameters found for impacts in {self.datacard_path}")
            arguments = [
                "combineTool.py",
                "-M", "Impacts",
                "-d", self.datacard_path,
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
                "--impactsJson", f"{os.path.join(temp_output_dir, 'Combine', self.outdir, self.fitFolderName, 'impact', 'impacts', 'impacts.json')}",
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
            named_params = _list_modelconfig_nuisances(self.datacard_path, ["r"], exclude_expr=exclude_expr)
            if not named_params:
                raise RuntimeError(f"No nuisance parameters found for impacts in {self.datacard_path}")
            arguments = [
                "combineTool.py",
                "-M", "Impacts",
                "-d", self.datacard_path,
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
                "-d", self.datacard_path,
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
            named_params = _list_modelconfig_nuisances(self.datacard_path, poi_list, exclude_expr=exclude_expr)
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


class AsimovCovCorrHesse(CombineTask, law.LocalWorkflow, HTCondorWorkflow, SlurmWorkflow): #(law.Task): #(CombineTask, HTCondorWorkflow, law.LocalWorkflow):

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

            
        if self.variable == "":
            output = []
        else:
            # output = [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName)]
            output = [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'hesse')]
            
            
            output += [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'hesse', f'robustHessefirstStep.root')]
            output += [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'hesse', f'multidimfitfirstStep.root')]
            output += [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'hesse', f'higgsCombinefirstStep.MultiDimFit.mH{HIGGS_MASS}.root')]
        
        outputFileTargets = []
                
        for _, current_output_path in enumerate(output):
            outputFileTargets.append(law.LocalFileTarget(current_output_path))
            
        # print(outputFileTargets)

        return outputFileTargets

    def run(self):
       
        if self.variable == '':
            print("Running AsimovCovCorrHesse for inclusive does not make sense. Please specify a variable.")
            exit(1)
            
                  
        config = self.get_input_config()
        output_dir = self.get_output_dir() 
            

        
        cwd = os.getcwd()
        
        execute_command([f'mkdir -p {output_dir}/Combine/{self.outdir}/{self.fitFolderName}/hesse'], shell=True)
        os.chdir(os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'hesse'))

        # Determine which discrete pdfindex categories actually exist in the workspace.
        # This is important when some bins use merged categories (e.g. *_catMerged_*)
        # and therefore do not define the usual *_cat0/cat1/cat2_* RooCategories.
        pdf_indices = combineVariableDict(self.variable, self.year)['pdfIndeces']
        cache_key = (self.datacard_path,)
        if cache_key not in _PDFINDEX_CACHE and os.path.exists(self.datacard_path):
            datacard_pdf_indices = extract_pdf_indices(self.datacard_path)
            if datacard_pdf_indices:
                _PDFINDEX_CACHE[cache_key] = datacard_pdf_indices
        if cache_key in _PDFINDEX_CACHE:
            pdf_indices = _PDFINDEX_CACHE[cache_key]
        arguments = [
            "combine",
            "-M", "MultiDimFit",
            self.datacard_path,
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
        
class AsimovCovCorr(CombineTask, law.LocalWorkflow, HTCondorWorkflow, SlurmWorkflow): #(law.Task): #(CombineTask, HTCondorWorkflow, law.LocalWorkflow):
    noPreliminary = law.Parameter(default=False, description="Flag, if final plot should bear the Preliminary.")


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

            
        # output = [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName)]
        output = [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'hesse', 'Plots')]

        output += [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'hesse', 'Plots', f'corrMatrix_{self.variable}_syst.pdf')]
        output += [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'hesse', 'Plots', f'corrMatrix_{self.variable}_syst.png')]
        output += [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'hesse', 'Plots', f'covMatrix_{self.variable}_syst.pdf')]
        output += [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'hesse', 'Plots', f'covMatrix_{self.variable}_syst.png')]
        
        outputFileTargets = []
                
        for _, current_output_path in enumerate(output):
            outputFileTargets.append(law.LocalFileTarget(current_output_path))
            
        # print(outputFileTargets)

        return outputFileTargets

    def run(self):
       
        if self.variable == '':
            # Does not make sense inclusively
            return True
            
                  
        config = self.get_input_config()
        output_dir = self.get_output_dir() 
            

        
        if self.batch_flavor == "slurm/psi":
            # Have to use /scratch/batch_username/ for slurm/psi
            if "/work" in output_dir:
                execute_command([f'mkdir -p {output_dir}/Combine/{self.outdir}/{self.fitFolderName}/hesse/Plots'], shell=True)
            else:   
                execute_command([f'xrdfs root://t3dcachedb03.psi.ch:1094/ mkdir -p {output_dir}/Combine/{self.outdir}/{self.fitFolderName}/hesse/Plots'], shell=True)

            os.environ["TARGET_PATH"] = f"/scratch/{os.environ['USER']}/{os.environ['SLURM_JOB_ID']}"
            execute_command([f'mkdir -p $TARGET_PATH/Combine/{self.outdir}/{self.fitFolderName}/hesse/Plots'], shell=True)

            if "/work" in output_dir:
                slurm_copy_command = [
                    'cp', '-rf',
                    f'{output_dir}/Combine/{self.outdir}/{self.fitFolderName}/hesse',
                    f'{os.environ["TARGET_PATH"]}/Combine/{self.outdir}/{self.fitFolderName}/'
                ]
            else:
                slurm_copy_command = [
                    'xrdcp', '-rf',
                    'root://t3dcachedb03.psi.ch:1094//'+ f'{output_dir}/Combine/{self.outdir}/{self.fitFolderName}/hesse',
                    f'{os.environ["TARGET_PATH"]}/Combine/{self.outdir}/{self.fitFolderName}/'
                ]
            execute_command(slurm_copy_command)
            temp_output_dir = os.environ["TARGET_PATH"]
            # output_dir = os.path.join(os.environ["TARGET_PATH"], 'Combine', self.fitFolderName, 'hesse', 'Plots')
        else:
            execute_command([f'mkdir -p {output_dir}/Combine/{self.outdir}/{self.fitFolderName}/hesse/Plots'], shell=True)
            temp_output_dir = output_dir
            # output_dir = os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'hesse', 'Plots')            
        
        cwd = os.getcwd()
        os.chdir(os.path.join(os.environ["ANALYSIS_PATH"], 'Plots'))

        arguments = [
            "python3",
            f"{os.path.join(os.environ['ANALYSIS_PATH'], 'Plots', 'makeCorrMatrix.py')}",
            "--inputJson", f"{os.path.join(os.environ['ANALYSIS_PATH'], 'Plots', 'inputs_robustHesse.json')}",
            "--mode", f"{self.variable}",
            "--input", f"{os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'hesse', f'robustHessefirstStep.root')}",
            "--output", f"{os.path.join(temp_output_dir, 'Combine', self.outdir, self.fitFolderName, 'hesse', 'Plots')}",
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
            "--input", f"{os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'hesse', f'robustHessefirstStep.root')}",
            "--output", f"{os.path.join(temp_output_dir, 'Combine', self.outdir, self.fitFolderName, 'hesse', 'Plots')}",
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

class UnblindedFitFirstStep(CombineTask, law.LocalWorkflow, HTCondorWorkflow, SlurmWorkflow): #(law.Task): #(CombineTask, HTCondorWorkflow, law.LocalWorkflow):

    # def requires(self):
    def workflow_requires(self):
        workflow_reqs = super().workflow_requires()

        tasks = {}

        if workflow_reqs:
            tasks.update(workflow_reqs)
        
        config = self.get_input_config()
        output_dir = self.get_output_dir()

        fitConfig = config["combine_fit"] 
 
        tasks["RunT2WS"] = RunText2Workspace.req(
            self,
            output_dir=output_dir,
            years=self.year)        

        return tasks
    
    def create_branch_map(self):
        if self.variable == '':
            param_list = ["r"]
        elif self.variable == 'MH':
            param_list = ["MH"]
        else:
            param_list = combineVariableDict(self.variable, self.year)['paramStrNoOne']
        branch_map = {i: current_cat for i, current_cat in enumerate(param_list)}
        return branch_map

    def output(self):
        current_cat = self.branch_data
        
        config = self.get_input_config()
        output_dir = self.get_output_dir()

        output = {}
        
        output['scan_folder'] = law.LocalFileTarget(os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'dataFit'))
        output['first_step_root'] = law.LocalFileTarget(os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'dataFit', f'higgsCombineDataPostFitBestFit_{current_cat}.MultiDimFit.mH{HIGGS_MASS}.root'))

        return output

    def run(self):
        current_cat = self.branch_data
        

                    
        config = self.get_input_config()

        # default values, if not set in config
        cminApproxPreFitTolerance = config.get("combine_fit", {}).get("cminApproxPreFitTolerance", 0.01)
        rMin = config.get("combine_fit", {}).get("rMin", 0.7)
        rMax = config.get("combine_fit", {}).get("rMax", 1.6)

        output_dir = self.get_output_dir()
        
        cwd = os.getcwd()
        

        fit_dir = os.path.join(
            output_dir, "Combine", self.outdir, self.fitFolderName, "dataFit"
        )
        os.makedirs(fit_dir, exist_ok=True)
        os.chdir(fit_dir)

        if self.variable == '':
            arguments = [                "combine",
                "-M", "MultiDimFit",
                "-d", self.datacard_path,
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
            
        elif self.variable == 'MH':
            arguments = [
                "combine", "-M", "MultiDimFit", 
                "-d", self.datacard_path,
                "-P", "MH",
                "--redefineSignalPOIs", "MH",
                "--freezeParameters", "r",
                "--setParameters", f"r=1,MH={HIGGS_MASS}",
                "--setParameterRanges", config['combine_fit']['setParameterRange'],
                "-m", f"{HIGGS_MASS}",
                "-n", f"DataPostFitBestFit_{current_cat}",
                "--algo", "singles", 
                "--floatOtherPOIs", "1", "--saveWorkspace", "--saveFitResult",
                "--cminDefaultMinimizerStrategy=0",
                "--cminFallbackAlgo", "Minuit2,Migrad,1:10",
                "--X-rtd", "MINIMIZER_freezeDisassociatedParams",
                "--X-rtd", "MINIMIZER_multiMin_hideConstants",
                "--X-rtd", "MINIMIZER_multiMin_maskConstraints",
                "--X-rtd", "MINIMIZER_multiMin_maskChannels=2",
            ]
            print(' '.join(arguments))
            subprocess.run(arguments, check=True, text=True)

        else:         
            arguments = [
                "combine",
                "-M", "MultiDimFit",
                "-d", self.datacard_path,
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

        os.chdir(cwd)


        

class UnblindedCovCorrHesse(CombineTask, law.LocalWorkflow, HTCondorWorkflow, SlurmWorkflow): #(law.Task): #(CombineTask, HTCondorWorkflow, law.LocalWorkflow):

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
            output = []
        else:
            output = [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'hesse')]

            output += [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'hesse', f'robustHessefirstStep_data.root')]
            output += [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'hesse', f'multidimfitfirstStep_data.root')]
            output += [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'hesse', f'higgsCombinefirstStep_data.MultiDimFit.mH{HIGGS_MASS}.root')]
            
        outputFileTargets = []
                
        for _, current_output_path in enumerate(output):
            outputFileTargets.append(law.LocalFileTarget(current_output_path))

        return outputFileTargets

    def run(self):
       
        if self.variable == '':
            print("Running UnblindedCovCorrHesse for inclusive does not make sense. Please specify a variable.")
            exit(1)
            
                  
        config = self.get_input_config()
        output_dir = self.get_output_dir() 
            


        cwd = os.getcwd()
        
        execute_command([f'mkdir -p {output_dir}/Combine/{self.outdir}/{self.fitFolderName}/hesse'], shell=True)
        os.chdir(os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'hesse'))

        arguments = [
            "combine",
            "-M", "MultiDimFit",
            self.datacard_path,
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
            
        os.chdir(cwd)
        
class UnblindedCovCorr(CombineTask, law.LocalWorkflow, HTCondorWorkflow, SlurmWorkflow): #(law.Task): #(CombineTask, HTCondorWorkflow, law.LocalWorkflow):
    noPreliminary = law.Parameter(default=False, description="Flag, if final plot should bear the Preliminary.")

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

            
        # output = [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName)]
        output = [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'hesse', 'Plots', 'data')]

        output += [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'hesse', 'Plots', 'data', f'corrMatrix_{self.variable}_syst_obs.pdf')]
        output += [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'hesse', 'Plots', 'data', f'corrMatrix_{self.variable}_syst_obs.png')]
        output += [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'hesse', 'Plots', 'data', f'covMatrix_{self.variable}_syst_obs.pdf')]
        output += [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'hesse', 'Plots', 'data', f'covMatrix_{self.variable}_syst_obs.png')]
        
        outputFileTargets = []
                
        for _, current_output_path in enumerate(output):
            outputFileTargets.append(law.LocalFileTarget(current_output_path))
            
        # print(outputFileTargets)

        return outputFileTargets

    def run(self):
       
        if self.variable == '':
            print("Running UnblindedCovCorrHesse for inclusive does not make sense. Please specify a variable.")
            exit(1)
        
                  
        config = self.get_input_config()
        output_dir = self.get_output_dir() 

        if self.batch_flavor == "slurm/psi":
            # Have to use /scratch/batch_username/ for slurm/psi
            if "/work" in output_dir:
                execute_command([f'mkdir -p {output_dir}/Combine/{self.outdir}/{self.fitFolderName}/hesse/Plots/data'], shell=True)
            else:   
                execute_command([f'xrdfs root://t3dcachedb03.psi.ch:1094/ mkdir -p {output_dir}/Combine/{self.outdir}/{self.fitFolderName}/hesse/Plots/data'], shell=True)

            os.environ["TARGET_PATH"] = f"/scratch/{os.environ['USER']}/{os.environ['SLURM_JOB_ID']}"
            execute_command([f'mkdir -p $TARGET_PATH/Combine/{self.outdir}/{self.fitFolderName}/hesse/Plots/data'], shell=True)

            if "/work" in output_dir:
                slurm_copy_command = [
                    'cp', '-rf',
                    f'{output_dir}/Combine/{self.outdir}/{self.fitFolderName}/hesse',
                    f'{os.environ["TARGET_PATH"]}/Combine/{self.outdir}/{self.fitFolderName}/'
                ]
            else:
                slurm_copy_command = [
                    'xrdcp', '-rf',
                    'root://t3dcachedb03.psi.ch:1094//'+ f'{output_dir}/Combine/{self.outdir}/{self.fitFolderName}/hesse',
                    f'{os.environ["TARGET_PATH"]}/Combine/{self.outdir}/{self.fitFolderName}/'
                ]
            execute_command(slurm_copy_command)
            # output_dir = os.path.join(os.environ["TARGET_PATH"], 'Combine', self.fitFolderName, 'hesse', 'Plots', 'data')
            temp_output_dir = os.environ["TARGET_PATH"]
        else:
            execute_command([f'mkdir -p {output_dir}/Combine/{self.outdir}/{self.fitFolderName}/hesse/Plots/data'], shell=True)
            temp_output_dir = output_dir
            # output_dir = os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'hesse', 'Plots', 'data')     
        
        cwd = os.getcwd()
        os.chdir(os.path.join(os.environ["ANALYSIS_PATH"], 'Plots'))

        arguments = [
            "python3",
            f"{os.path.join(os.environ['ANALYSIS_PATH'], 'Plots', 'makeCorrMatrix.py')}",
            "--inputJson", f"{os.path.join(os.environ['ANALYSIS_PATH'], 'Plots', 'inputs_robustHesse.json')}",
            "--mode", f"{self.variable}",
            "--input", f"{os.path.join(temp_output_dir, 'Combine', self.outdir, self.fitFolderName, 'hesse', f'robustHessefirstStep_data.root')}",
            "--output", f"{os.path.join(temp_output_dir, 'Combine', self.outdir, self.fitFolderName, 'hesse', 'Plots', 'data')}",
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
            "--input", f"{os.path.join(temp_output_dir, 'Combine', self.outdir, self.fitFolderName, 'hesse', f'robustHessefirstStep_data.root')}",
            "--output", f"{os.path.join(temp_output_dir, 'Combine', self.outdir, self.fitFolderName, 'hesse', 'Plots', 'data')}",
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
        
        
class UnblindedImpactFirstStep(CombineTask, law.LocalWorkflow, HTCondorWorkflow, SlurmWorkflow): #(law.Task): #(CombineTask, HTCondorWorkflow, law.LocalWorkflow):
    # def requires(self):
    def workflow_requires(self):
        workflow_reqs = super().workflow_requires()

        tasks = {}

        if workflow_reqs:
            tasks.update(workflow_reqs)
        
        config = self.get_input_config()
        output_dir = self.get_output_dir()

        impactConfig = config["combine_impacts"]

        tasks["RunT2WS"] = RunText2Workspace.req(
            self,
            output_dir=output_dir, 
            years=self.year
            )

        return tasks

    def create_branch_map(self):
        branch_map = {i: current_branch for i, current_branch in enumerate([0])}
        return branch_map

    def output(self):        
        config = self.get_input_config()
        output_dir = self.get_output_dir()

            
        output = [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'impact', 'unblinded')]

        output += [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'impact', 'unblinded', f'higgsCombine_initialFit_Test.MultiDimFit.mH{HIGGS_MASS}.root')]
        
        outputFileTargets = []
                
        for _, current_output_path in enumerate(output):
            outputFileTargets.append(law.LocalFileTarget(current_output_path))

        return outputFileTargets

    def run(self):
       

                  
        config = self.get_input_config()

        # default values, if not set in config
        cminApproxPreFitTolerance = config.get("combine_impacts", {}).get("cminApproxPreFitTolerance", 0.01)
        setParameters = config.get("combine_impacts", {}).get("setParameters", 1.000)

        if self.output_dir == '':
            output_dir = config['outputFolder']
        else:
            output_dir = self.output_dir  
            

        
        cwd = os.getcwd()
        execute_command([f'mkdir -p {output_dir}/Combine/{self.outdir}/{self.fitFolderName}/impact/unblinded'], shell=True)
        os.chdir(os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'impact', 'unblinded'))
                    
        common_arguments = [
            "-d", self.datacard_path,
            "-m", f"{HIGGS_MASS}",
            "--robustFit", "1",
            "--cminDefaultMinimizerStrategy=0",
            "--cminFallbackAlgo", "Minuit2,Migrad,1:10",
            "--X-rtd", "MINIMIZER_freezeDisassociatedParams",
            "--X-rtd", "MINIMIZER_multiMin_hideConstants",            "--X-rtd", "MINIMIZER_multiMin_maskConstraints",
            "--X-rtd", "MINIMIZER_multiMin_maskChannels=2",
            "--cminApproxPreFitTolerance", f"{cminApproxPreFitTolerance}",
        ]

        if self.variable in ("", "MH"):
            arguments = [
                "combineTool.py",
                "-M", "Impacts",
                *common_arguments,
                "--doInitialFit",
            ]

            if self.variable == "":
                # Inclusive signal-strength fit: profile r and keep the Higgs
                # mass fixed.  Accept either a bare configured value or a full
                # Combine assignment such as "r=1".
                initial_r = str(setParameters)
                if "=" not in initial_r:
                    initial_r = f"r={initial_r}"
                arguments += [
                    "--freezeParameters", "MH",
                    "--setParameters", initial_r,
                ]
            else:
                # Higgs-mass fit: MH is the POI and must float, while r is held
                # at the nominal signal strength.
                mass_range = config.get("combine_fit", {}).get(
                    "setParameterRange", "MH=124,126"
                )
                arguments += [
                    "--redefineSignalPOIs", "MH",
                    "--freezeParameters", "r",
                    "--setParameters", f"r=1,MH={HIGGS_MASS}",
                    "--setParameterRanges", mass_range,
                ]
        else:
            pois = combineVariableDict(
                self.variable, self.year
            )["paramStrNoOne"]
            arguments = [
                "combine",
                "-M", "MultiDimFit",
                *common_arguments,
                "--algo", "singles",
                "--redefineSignalPOIs", ",".join(pois),
                "--freezeParameters", "MH",
                "-n", "_initialFit_Test",
            ]

        print("UnblindedImpactFirstStep: " + " ".join(arguments))
        try:
            result = subprocess.run(
                arguments,
                check=True,
                text=True,
                capture_output=True,
            )
            print("Script output:", result.stdout)
            print("Script executed successfully.")
        except subprocess.CalledProcessError as e:
            print("Error executing script:", e.stderr)
            raise
            
        os.chdir(cwd)
        
class UnblindedImpactSecondStep(CombineTask, law.LocalWorkflow, HTCondorWorkflow, SlurmWorkflow): #(law.Task): #(CombineTask, HTCondorWorkflow, law.LocalWorkflow):
    # def requires(self):
    def workflow_requires(self):
        workflow_reqs = super().workflow_requires()

        tasks = {}

        if workflow_reqs:
            tasks.update(workflow_reqs)
        
        config = self.get_input_config()
        output_dir = self.get_output_dir()

        impactConfig = config["combine_impacts"]
            
        tasks["UnblindedImpactFirstStep"] = UnblindedImpactFirstStep.req(
            self,
            output_dir=output_dir, 
            # workflow=impactConfig["execution"], 
            # slurm_partition=impactConfig['batchPartition'], 
            # slurm_memory=impactConfig['batchMemory'], 
            # slurm_max_runtime=impactConfig['batchMaxRuntime'], 
            # htcondor_partition=impactConfig['batchPartition'], 
            # htcondor_memory=impactConfig['batchMemory'], 
            # htcondor_max_runtime=impactConfig['batchMaxRuntime']
            )
        
        return tasks

    def create_branch_map(self):
        

        config = self.get_input_config()
        output_dir = self.get_output_dir() 
            

    
        if self.variable == '':
            poiList = ["r"]
        elif self.variable == 'MH':
            # r is fixed in the MH impacts workflow, so exclude it from the
            # nuisance branches as well as excluding the actual POI.
            poiList = ["MH", "r"]
        else:
            poiList = combineVariableDict(self.variable, self.year)['paramStrNoOne']

        # Use only constrained ModelConfig nuisances, matching the Asimov
        # workflow.  All-free-PDF parameters also pull in env_pdf/shapeBkg
        # parameters, which are not standard +/-1 sigma impact nuisances.
        exclude_expr = config.get("combine_impacts", {}).get("exclude", "")
        paramList = _list_modelconfig_nuisances(
            self.datacard_path, poiList, exclude_expr=exclude_expr
        )
        
        branch_map = {i: current_param for i, current_param in enumerate(paramList)}
        return branch_map

    def output(self):        
        current_param = self.branch_data
        
        config = self.get_input_config()
        output_dir = self.get_output_dir()

            
        # output = [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName)]
        output = [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'impact', 'unblinded')]

        output += [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'impact', 'unblinded', f'higgsCombine_paramFit_Test_{current_param}.MultiDimFit.mH{HIGGS_MASS}.root')]
        
        outputFileTargets = []
                
        for _, current_output_path in enumerate(output):
            outputFileTargets.append(law.LocalFileTarget(current_output_path))
            
        # print(outputFileTargets)

        return outputFileTargets

    def complete(self):
        if self.branch < 0:
            return super().complete()
        output_path = self.output()[-1].path
        if not os.path.isfile(output_path) or os.path.getsize(output_path) < 1000:
            return False
        try:
            with uproot.open(output_path) as root_file:
                return (
                    "limit" in root_file
                    and root_file["limit"].num_entries >= 3
                )
        except Exception as exc:
            print(f"Invalid impact output '{output_path}': {exc}")
            return False

    def workflow_complete(self):
        """Validate every branch output instead of checking file existence only."""
        return all(
            branch_task.complete()
            for branch_task in self.get_branch_tasks().values()
        )

    def run(self):
        current_param = self.branch_data
       

                  
        #Load central config file
        config = self.get_input_config()

        # default values, if not set in config
        cminApproxPreFitTolerance = config.get("combine_impacts", {}).get("cminApproxPreFitTolerance", 0.01)
        stepSize = config.get("combine_impacts", {}).get("stepSize", 0.05)
        setCrossingTolerance = config.get("combine_impacts", {}).get("setCrossingTolerance", 0.00005)

        output_dir = self.get_output_dir() 
            

        
        cwd = os.getcwd()

        execute_command([f'mkdir -p {output_dir}/Combine/{self.outdir}/{self.fitFolderName}/impact/unblinded'], shell=True)
        os.chdir(os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'impact', 'unblinded'))

        if self.variable == '':            
            initial_fit = os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'impact', 'unblinded', f'higgsCombine_initialFit_Test.MultiDimFit.mH{HIGGS_MASS}.root')
            
            f = ROOT.TFile(initial_fit)
            tree = f.Get("limit")
            
            if not tree:
                print("Error: Tree 'limit' not found in the file.")
                exit(1)
            # Access the branch 'r_YH_2p0_2p5' and get its first value
            if hasattr(tree, 'r'):
                tree.GetEntry(0)  # Load the first entry
                poi_bf_value = getattr(tree, 'r')  # Access the branch value                poi_bf_string = f'r={poi_bf_value}'
                print(f"First value of branch 'r': {poi_bf_value}")
            else:
                print("Error: Branch 'r' not found in the tree.")
                exit(1)

            arguments = [
                "combine",
                "-M", "MultiDimFit",
                "-d", self.datacard_path,
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

        elif self.variable == "MH":
            initial_fit = os.path.join(
                output_dir,
                'Combine',
                self.outdir,
                self.fitFolderName,
                'impact',
                'unblinded',
                f'higgsCombine_initialFit_Test.MultiDimFit.mH{HIGGS_MASS}.root',
            )
            f = ROOT.TFile(initial_fit)
            tree = f.Get("limit")
            if not tree or not hasattr(tree, "MH"):
                raise RuntimeError(
                    f"Initial impact fit has no readable MH branch: {initial_fit}"
                )
            tree.GetEntry(0)
            mh_bf = getattr(tree, "MH")
            f.Close()

            mass_range = config.get("combine_fit", {}).get(
                "setParameterRange", "MH=124,126"
            )
            arguments = [
                "combine",
                "-M", "MultiDimFit",
                "-d", self.datacard_path,
                "--algo", "impact",
                "--redefineSignalPOIs", "MH",
                "--setParameters", f"r=1,MH={mh_bf}",
                "--setParameterRanges", mass_range,
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
                "--cminApproxPreFitTolerance", f"{cminApproxPreFitTolerance}",
                "--stepSize", f"{stepSize}",
                "--setCrossingTolerance", f"{setCrossingTolerance}",
                "--robustHesse", "1",
            ]
            print("UnblindedImpactSecondStep (MH): " + " ".join(arguments))
            try:
                result = subprocess.run(
                    arguments, check=True, text=True, capture_output=True
                )
                print("Script output:", result.stdout)
                print("Script executed successfully.")
            except subprocess.CalledProcessError as e:
                print("Error executing script:", e.stderr)
                raise

        else:

            initial_fit = os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'impact', 'unblinded', f'higgsCombine_initialFit_Test.MultiDimFit.mH{HIGGS_MASS}.root')

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
                "-d", self.datacard_path,
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
            
        os.chdir(cwd)
        
class UnblindedImpactThirdStep(CombineTask, law.LocalWorkflow, HTCondorWorkflow, SlurmWorkflow): #(law.Task): #(CombineTask, HTCondorWorkflow, law.LocalWorkflow):


    # def requires(self):
    def workflow_requires(self):
        workflow_reqs = super().workflow_requires()

        tasks = {}

        if workflow_reqs:
            tasks.update(workflow_reqs)
        
        
        config = self.get_input_config()
        output_dir = self.get_output_dir()

        impactConfig = config["combine_impacts"]
            
        tasks["UnblindedImpactSecondStep"] = UnblindedImpactSecondStep.req(
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


        output = []
        if self.variable in ['', 'MH']:
            cat = 'MH' if self.variable == 'MH' else 'r'
            output += [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'impact', 'unblinded', 'impacts')]
            output += [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'impact', 'unblinded', 'impacts', f'impacts_unblinded.pdf')]
            output += [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'impact', 'unblinded', 'impacts', 'impacts_corrected_dropBkgModelParams.json')]
            
        else:
            for cat in combineVariableDict(self.variable, self.year)['paramStrNoOne']:
                output += [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'impact', 'unblinded', 'impacts')]
                output += [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'impact', 'unblinded', 'impacts', f'impacts_unblinded_{cat}.pdf')]
                output += [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'impact', 'unblinded', 'impacts', 'impacts_corrected_dropBkgModelParams.json')]
                
        output += [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'impact', 'unblinded', 'impacts', f'impacts.json')]
        
        outputFileTargets = []
                
        for _, current_output_path in enumerate(output):
            outputFileTargets.append(law.LocalFileTarget(current_output_path))
            
        # print(outputFileTargets)

        return outputFileTargets

    def run(self):
       
                  
        config = self.get_input_config()
        output_dir = self.get_output_dir() 
            


        cwd = os.getcwd()

        execute_command([f'mkdir -p {output_dir}/Combine/{self.outdir}/{self.fitFolderName}/impact/unblinded/impacts'], shell=True)
        os.chdir(os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'impact', 'unblinded'))
        temp_output_dir = output_dir

        if self.variable in ('', 'MH'):
            is_mh = self.variable == 'MH'
            poi_list = ["MH", "r"] if is_mh else ["r"]
            nuisance_parameters = _list_modelconfig_nuisances(
                self.datacard_path,
                poi_list,
                exclude_expr=config["combine_impacts"].get("exclude", ""),
            )
            if not nuisance_parameters:
                raise RuntimeError(
                    f"No constrained nuisance parameters found in '{self.datacard_path}'"
                )
            excluded_parameters = [
                item.strip()
                for item in config["combine_impacts"].get("exclude", "").split(",")
                if item.strip()            ]
            if is_mh and "r" not in excluded_parameters:
                excluded_parameters.append("r")
            arguments = [
                "combineTool.py",
                "-M", "Impacts",
                "-d", self.datacard_path,
                "-m", f"{HIGGS_MASS}",
                "-o", "impacts/impacts.json",
                "--named", ",".join(nuisance_parameters),
            ]
            if is_mh:
                arguments += [
                    "--redefineSignalPOIs", "MH",
                    "--freezeParameters", "r",
                ]
            if excluded_parameters:
                arguments += ["--exclude", ",".join(excluded_parameters)]
            command = arguments
            print(' '.join(command))
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
                "--impactsJson", f"{os.path.join(temp_output_dir, 'Combine', self.outdir, self.fitFolderName, 'impact', 'unblinded', 'impacts', 'impacts.json')}",
                "--frozenParam", "r" if is_mh else "MH",
                "--dropBkgModelParams"
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
                
            arguments = [
                "plotImpacts.py",
                "-i", f"{os.path.join(temp_output_dir, 'Combine', self.outdir, self.fitFolderName, 'impact', 'unblinded', 'impacts', 'impacts_corrected_dropBkgModelParams.json')}",
                "-o", "impacts/impacts_unblinded",
            ]
            if is_mh:
                arguments += [
                    "--POI", "MH",
                    "--translate", os.path.join(
                        os.environ['ANALYSIS_PATH'], 'Combine', 'pois.json'
                    ),
                ]
            if config['combine_impacts']['not_show_POI']:
                arguments.append("--blind")
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
                "-d", self.datacard_path,
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
                raise
                
            arguments = [
                "python3",
                f"{os.path.join(os.environ['ANALYSIS_PATH'], 'Plots', 'correctImpacts.py')}",
                "--impactsJson", f"{os.path.join(temp_output_dir, 'Combine', self.outdir, self.fitFolderName, 'impact', 'unblinded', 'impacts', 'impacts.json')}",
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
                raise
                
            for cat in combineVariableDict(self.variable, self.year)['paramStrNoOne']:
                arguments = [
                    "plotImpacts.py",
                    "-i", f"{os.path.join(temp_output_dir, 'Combine', self.outdir, self.fitFolderName, 'impact', 'unblinded', 'impacts', 'impacts_corrected_dropBkgModelParams.json')}",
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
                    raise
            
        os.chdir(cwd)
        
        
class MggBestFit(CombineTask, law.LocalWorkflow, HTCondorWorkflow, SlurmWorkflow): #(law.Task): #(CombineTask, HTCondorWorkflow, law.LocalWorkflow):

    # def requires(self):
    def workflow_requires(self):
        workflow_reqs = super().workflow_requires()

        tasks = {}

        if workflow_reqs:
            tasks.update(workflow_reqs)
        
        config = self.get_input_config()
        output_dir = self.get_output_dir()

        toyConfig = config["combine_mggToys"]

        tasks["RunT2WS"] = RunText2Workspace.req(
            self,
            output_dir=output_dir, 
            years=self.year,
            # workflow=toyConfig["execution"], version=self.variable if self.variable != "" else "inclusive", slurm_partition=toyConfig.get('batchPartition', None), slurm_memory=toyConfig.get('batchMemory', None), slurm_max_runtime=toyConfig.get('batchMaxRuntime', None), htcondor_partition=toyConfig.get('batchPartition', None), htcondor_memory=toyConfig.get('batchMemory', None), htcondor_max_runtime=toyConfig.get('batchMaxRuntime', None))
        )
        return tasks

    def create_branch_map(self):
            
        if self.variable in ["", "tuto"]:
            cat_list = ["r"]
        elif self.variable == "MH":
            cat_list = ["MH"]
        else:
            cat_list = combineVariableDict(self.variable, self.year)['paramStrNoOne']

        branch_map = {i: current_cat for i, current_cat in enumerate(cat_list)}
        return branch_map

    def output(self):        
        cat = self.branch_data
                
        config = self.get_input_config()
        output_dir = self.get_output_dir()

            
        output = [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'postFit')]
        output += [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'postFit', f'SplusBModels_{cat}')]
        output += [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'postFit', f'SplusBModels_{cat}', f'higgsCombine_bestfit_syst_obs_{cat}.MultiDimFit.mH{HIGGS_MASS}.root')]
        
        outputFileTargets = []
                
        for _, current_output_path in enumerate(output):
            outputFileTargets.append(law.LocalFileTarget(current_output_path))
            
        # print(outputFileTargets)

        return outputFileTargets

    def run(self):
        cat = self.branch_data
       

                  
        config = self.get_input_config()
        output_dir = self.get_output_dir() 
            

             
        cwd = os.getcwd()

        execute_command([f'mkdir -p {output_dir}/Combine/{self.outdir}/{self.fitFolderName}/postFit/SplusBModels_{cat}'], shell=True)
        os.chdir(os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'postFit', f'SplusBModels_{cat}'))

        arguments = [
            "combine",
            "--floatOtherPOIs", "1",
            "-P", f"{cat}",
            "--saveInactivePOI", "1",
            "--saveWorkspace",
            "--saveSpecifiedNuis", "all",
            "--cminDefaultMinimizerStrategy", "0",
            "--X-rtd", "MINIMIZER_freezeDisassociatedParams",
            "--X-rtd", "MINIMIZER_multiMin_hideConstants",
            "--X-rtd", "MINIMIZER_multiMin_maskConstraints",
            "--X-rtd", "MINIMIZER_multiMin_maskChannels=2",            "-M", f"{config['combine_mggToys']['loadSnapshot']}",
            "-m", f"{HIGGS_MASS}",
            "-d", self.datacard_path,
            "-n", f"_bestfit_syst_obs_{cat}"
        ]

        if self.variable == "MH":
            arguments += ["--redefineSignalPOIs", "MH"]            
            arguments += ["--freezeParameters", "r"]
            arguments += ["--setParameters", f"r=1,MH={HIGGS_MASS}"]
        else:
            arguments += ["--freezeParameters", "MH"]
            arguments += ["--setParameters", f"MH={HIGGS_MASS}"]
        command = arguments
        # print(command)
        try:
            result = subprocess.run(command, check=True, text=True, capture_output=True)
            print("Script output:", result.stdout)
            print("Script executed successfully.")
        except subprocess.CalledProcessError as e:
            print("Error executing script:", e.stderr)

        
        os.chdir(cwd)
        
def _valid_gof_output(path, minimum_entries=1):
    """Return whether a GOF ROOT output has a readable, populated limit tree."""
    if not os.path.isfile(path) or os.path.getsize(path) < 1000:
        return False
    try:
        with uproot.open(path) as root_file:
            return "limit" in root_file and root_file["limit"].num_entries >= minimum_entries
    except Exception as exc:
        print(f"Invalid GOF output '{path}': {exc}")
        return False


class UnblindedGoodnessOfFitData(CombineTask):
    """Compute the saturated GOF statistic for the observed dataset."""

    mass = luigi.FloatParameter(default=125.0)
    mh = luigi.FloatParameter(default=125.38)
    parameter_ranges = law.Parameter(
        default="MH=122.0,128.0:r=0.0,2.0"
    )

    def requires(self):

        return RunText2Workspace.req(
            self,
            output_dir=self.get_output_dir(),
            years=self.year,
            )


    def _gof_dir(self):
        fit_folder = "runFits_mu_fiducial" if self.variable == "" else f"runFits_{self.variable}"
        return os.path.join(
            self.get_output_dir(), "Combine", self.outdir, fit_folder, "gof", "unblinded"
        )

    def _datacard_path(self):
        name = (
            f"Datacard_{self.year}.root"
            if self.variable == ""
            else f"Datacard_{self.variable}_{self.year}.root"
        )
        return os.path.join(self.get_output_dir(), "Combine", self.outdir, name)

    def output(self):
        mass_label = f"{self.mass:g}"
        return law.LocalFileTarget(
            os.path.join(
                self._gof_dir(),
                f"higgsCombine_data.GoodnessOfFit.mH{mass_label}.root",
            )
        )

    def complete(self):
        return _valid_gof_output(self.output().path)

    def run(self):
        os.makedirs(self._gof_dir(), exist_ok=True)
        arguments = [
            "combine",
            "-v", "1",
            "-M", "GoodnessOfFit",
            self._datacard_path(),
            "-m", f"{self.mass:g}",
            "--setParameters", f"MH={self.mh:g}",
            "--setParameterRanges", self.parameter_ranges,
            "--algo", "saturated",
            "-n", "_data",
        ]
        print("UnblindedGoodnessOfFitData: " + " ".join(arguments))
        subprocess.run(arguments, cwd=self._gof_dir(), check=True, text=True)


class UnblindedGoodnessOfFitToys(
    Task, law.LocalWorkflow, HTCondorWorkflow, SlurmWorkflow
):
    """Generate one saturated-GOF pseudo-experiment per workflow branch."""

    n_toys = luigi.IntParameter(default=500)
    toys_per_job = luigi.IntParameter(default=10)
    mass = luigi.FloatParameter(default=125.07)
    mh = luigi.FloatParameter(default=125.07)
    signal_strength = luigi.FloatParameter(default=1.0)
    parameter_ranges = law.Parameter(
        default="MH=122.0,128.0:r=0.0,2.0"
    )

    def workflow_requires(self):
        requirements = super().workflow_requires()
        requirements["RunT2WS"] = RunText2Workspace.req(
            self,
            output_dir=self.get_output_dir(),
            years=self.year,
        )
        return requirements

    def create_branch_map(self):
        return {index: index for index in range(math.ceil(self.n_toys / self.toys_per_job))}

    def _gof_dir(self):
        fit_folder = "runFits_mu_fiducial" if self.variable == "" else f"runFits_{self.variable}"
        return os.path.join(
            self.get_output_dir(), "Combine", self.outdir, fit_folder, "gof", "unblinded"
        )

    def _toy_dir(self):
        return os.path.join(self._gof_dir(), "toys")

    def _datacard_path(self):
        name = (
            f"Datacard_{self.year}.root"
            if self.variable == ""
            else f"Datacard_{self.variable}_{self.year}.root"
        )
        return os.path.join(self.get_output_dir(), "Combine", self.outdir, name)

    def output(self):
        return law.LocalFileTarget(
            os.path.join(self._toy_dir(), f"toy_{self.branch_data}.root")
        )

    def _toys_in_branch(self):
        first_toy = self.branch_data * self.toys_per_job
        return min(self.toys_per_job, self.n_toys - first_toy)

    def complete(self):
        if self.branch < 0:
            return super().complete()
        return _valid_gof_output(self.output().path, self._toys_in_branch())

    def run(self):
        toy = self.branch_data
        os.makedirs(self._toy_dir(), exist_ok=True)
        name = f"_{toy}_gen_step"
        pattern = os.path.join(
            self._toy_dir(),
            f"higgsCombine{name}.GoodnessOfFit.mH{self.mass:g}*.root",
        )
        # Random seed -1 adds a random suffix to the output filename.  Keep
        # track of files from earlier failed/retried branches so that only the
        # output produced by this invocation is selected below.
        files_before = set(glob.glob(pattern))
        arguments = [
            "combine",
            "-M", "GoodnessOfFit",
            self._datacard_path(),
            "-m", f"{self.mass:g}",
            "--setParameters",
            f"MH={self.mh:g},r={self.signal_strength:g}",
            "--setParameterRanges", self.parameter_ranges,
            "--algo", "saturated",
            "-t", f"{self._toys_in_branch()}",
            "--toysFrequentist",
            "-s", "-1",
            "-n", name,
        ]
        print("UnblindedGoodnessOfFitToys: " + " ".join(arguments))
        subprocess.run(arguments, cwd=self._toy_dir(), check=True, text=True)

        all_matching = set(glob.glob(pattern))
        generated = sorted(all_matching - files_before)
        if len(generated) != 1:
            raise RuntimeError(
                "Expected exactly one new GOF output from the current Combine "
                f"invocation matching '{pattern}', found new files {generated}; "
                f"all matching files are {sorted(all_matching)}"
            )
        shutil.move(generated[0], self.output().path)


class UnblindedGoodnessOfFit(CombineTask):
    """Merge GOF toys, collect the result and make the saturated-GOF plot."""

    n_toys = luigi.IntParameter(default=50)
    mass = luigi.FloatParameter(default=125.07)
    mh = luigi.FloatParameter(default=125.07)
    signal_strength = luigi.FloatParameter(default=1.0)
    parameter_ranges = law.Parameter(
        default="MH=122.0,128.0:r=0.0,2.0"
    )

    def _gof_dir(self):
        fit_folder = "runFits_mu_fiducial" if self.variable == "" else f"runFits_{self.variable}"
        return os.path.join(
            self.get_output_dir(), "Combine", self.outdir, fit_folder, "gof", "unblinded"
        )

    def requires(self):
        config = self.get_input_config()
        gof_config = config.get("combine_gof", {})
        workflow = gof_config.get("execution", "local")
        toys_per_job = gof_config.get("toysPerJob", 1)
        n_toys = gof_config.get("nToys", 50)
        workflow_kwargs = {"workflow": workflow, "toys_per_job": toys_per_job, "n_toys": n_toys}
        optional_workflow_parameters = {
            "slurm_partition": "batchPartition",
            "slurm_memory": "batchMemory",
            "slurm_max_runtime": "batchMaxRuntime",
            "htcondor_partition": "batchPartition",
            "htcondor_memory": "batchMemory",
            "htcondor_max_runtime": "batchMaxRuntime",
        }
        for task_parameter, config_key in optional_workflow_parameters.items():
            if config_key in gof_config:
                workflow_kwargs[task_parameter] = gof_config[config_key]
        return {
            "data": UnblindedGoodnessOfFitData.req(self),
            "toys": UnblindedGoodnessOfFitToys.req(self, **workflow_kwargs),
        }

    def output(self):
        return {
            "toys_root": law.LocalFileTarget(
                os.path.join(
                    self._gof_dir(),
                    f"higgsCombine_toys.GoodnessOfFit.mH{self.mass:g}.root",
                )
            ),
            "json": law.LocalFileTarget(os.path.join(self._gof_dir(), "gof.json")),
            "pdf": law.LocalFileTarget(os.path.join(self._gof_dir(), "gof_plot.pdf")),
            "png": law.LocalFileTarget(os.path.join(self._gof_dir(), "gof_plot.png")),
        }

    def run(self):
        os.makedirs(self._gof_dir(), exist_ok=True)
        gof_config = self.get_input_config().get("combine_gof", {})
        n_toys = int(gof_config.get("nToys", self.n_toys))
        toys_per_job = int(gof_config.get("toysPerJob", 1))
        n_toy_jobs = math.ceil(n_toys / toys_per_job)
        toy_inputs = [
            os.path.join(self._gof_dir(), "toys", f"toy_{toy}.root")
            for toy in range(n_toy_jobs)
        ]
        subprocess.run(
            ["hadd", "-f", self.output()["toys_root"].path] + toy_inputs,
            cwd=self._gof_dir(),
            check=True,
            text=True,
        )

        data_input = self.input()["data"].path
        command = [
                        "combineTool.py",
                        "-M", "CollectGoodnessOfFit",
                        "--input", data_input, self.output()["toys_root"].path,
                        "-m", str(self.mass),
                        "-o", self.output()["json"].path,
                    ]
        print(" ".join(command))
        subprocess.run(
            command,
            cwd=self._gof_dir(),
            check=True,
            text=True,
        )

        # The mh branch in Combine ROOT files is a float32.  Consequently,
        # CollectGoodnessOfFit can serialize 125.07 as e.g.
        # "125.06999969482422".  plotGof.py performs an exact string lookup,
        # so pass the key actually written to the JSON rather than the CLI
        # representation of the requested mass.
        with open(self.output()["json"].path) as gof_file:
            gof_payload = json.load(gof_file)
        if not gof_payload:
            raise RuntimeError(
                f"Collected GOF JSON is empty: {self.output()['json'].path}"
            )
        try:
            plot_mass = min(
                gof_payload,
                key=lambda key: abs(float(key) - float(self.mass)),
            )
        except ValueError as exc:
            raise RuntimeError(
                f"GOF JSON has no numerical mass key: {list(gof_payload)}"
            ) from exc

        subprocess.run(
            [
                "plotGof.py",
                self.output()["json"].path,
                "--statistic", "saturated",
                "--mass", plot_mass,
                "-o", "gof_plot",
            ],
            cwd=self._gof_dir(),
            check=True,
            text=True,
        )


class MggToyGeneration(CombineTask, law.LocalWorkflow, HTCondorWorkflow, SlurmWorkflow): #(law.Task): #(CombineTask, HTCondorWorkflow, law.LocalWorkflow):

    is_postfit = luigi.BoolParameter(
        default=False,
        description="Flag that signifies if toys are created for postfit mass distributions.",
    )

    # def requires(self):
    def workflow_requires(self):
        workflow_reqs = super().workflow_requires()

        tasks = {}

        if workflow_reqs:
            tasks.update(workflow_reqs)
        
        config = self.get_input_config()

        mggConfig = config['combine_mggToys']
        
        output_dir = self.get_output_dir()

        if self.is_postfit:
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
        toy, cat = self.branch_data
                
        output_dir = self.get_output_dir()

            
        if self.is_postfit:
            output = [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'postFit', f'SplusBModels_{cat}', 'toys')]
            output += [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'postFit', f'SplusBModels_{cat}', 'toys', 'filechecker')]
            output += [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'postFit', f'SplusBModels_{cat}', 'toys', 'filechecker', f'toy_{toy}_ok.txt')]
            output += [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'postFit', f'SplusBModels_{cat}', 'toys', f'toy_{toy}.root')]
        else:
            output = [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'preFit')]
            output += [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'preFit', f'SplusBModels_{cat}')]
            output += [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'preFit', f'SplusBModels_{cat}', 'toys')]
            output += [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'preFit', f'SplusBModels_{cat}', 'toys', 'filechecker')]
            output += [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'preFit', f'SplusBModels_{cat}', 'toys', 'filechecker', f'toy_{toy}_ok.txt')]
            output += [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'preFit', f'SplusBModels_{cat}', 'toys', f'toy_{toy}.root')]
        
        outputFileTargets = []
                
        for _, current_output_path in enumerate(output):
            outputFileTargets.append(law.LocalFileTarget(current_output_path))

        return outputFileTargets

    def run(self):
        toy, cat = self.branch_data
       

                  
        config = self.get_input_config()
        output_dir = self.get_output_dir() 

            

             
        cwd = os.getcwd()

        if self.is_postfit:

            os.environ["TARGET_PATH"] = f"/scratch/{os.environ['USER']}/{os.environ['SLURM_JOB_ID']}"

            execute_command([f'mkdir -p {output_dir}/Combine/{self.outdir}/{self.fitFolderName}/postFit/SplusBModels_{cat}/toys/filechecker'], shell=True)
            os.chdir(os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'postFit', f'SplusBModels_{cat}', 'toys'))
        
            best_fit = os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'postFit', f'SplusBModels_{cat}', f'higgsCombine_bestfit_syst_obs_{cat}.MultiDimFit.mH{HIGGS_MASS}.root')

            f = ROOT.TFile(best_fit)
            w = f.Get("w")
            w.loadSnapshot(config['combine_mggToys']['loadSnapshot'])
            poi_bf = w.var(cat).getVal()

            arguments = [
                "combine",
                best_fit,
                "-M", "GenerateOnly",                "-m", f"{HIGGS_MASS}",
                "--saveWorkspace",
                "--toysFrequentist",
                "--bypassFrequentistFit",
                "-t", "1",
                "-s", "-1",
                "-n", f"_{toy}_gen_step",
                "--setParameters", f"{cat}={poi_bf}",                "--snapshotName", f"{config['combine_mggToys']['loadSnapshot']}"
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
            toy_output_dir = output_dir

            source_pattern = os.path.join(toy_output_dir, 'Combine', self.outdir, self.fitFolderName, 'postFit', f'SplusBModels_{cat}', 'toys', f'higgsCombine_{toy}_gen_step*.root')
            destination = os.path.join(toy_output_dir, 'Combine', self.outdir, self.fitFolderName, 'postFit', f'SplusBModels_{cat}', 'toys', f'gen_{toy}.root')

            # Use glob to find files matching the source pattern
            source_files = glob.glob(source_pattern)

            if not source_files:
                print("No files found matching the pattern.")
            else:
                # Move each matched file to the destination
                for source_file in source_files:
                    try:
                        print(f"Moving {source_file} to {destination}")
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
            source_pattern = os.path.join(toy_output_dir, 'Combine', self.outdir, self.fitFolderName, 'postFit', f'SplusBModels_{cat}', 'toys', f'higgsCombine_{toy}_fit_step*.root')
            destination = os.path.join(toy_output_dir, 'Combine', self.outdir, self.fitFolderName, 'postFit', f'SplusBModels_{cat}', 'toys', f'fit_{toy}.root')

            # Use glob to find files matching the source pattern
            source_files = glob.glob(source_pattern)

            if not source_files:
                print("No files found matching the pattern.")
            else:
                # Move each matched file to the destination
                for source_file in source_files:
                    try:
                        print(f"Moving {source_file} to {destination}")
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
            source_pattern = os.path.join(toy_output_dir, 'Combine', self.outdir, self.fitFolderName, 'postFit', f'SplusBModels_{cat}', 'toys', f'higgsCombine_{toy}_throw_step*.root')
            destination = os.path.join(toy_output_dir, 'Combine', self.outdir, self.fitFolderName, 'postFit', f'SplusBModels_{cat}', 'toys', f'toy_{toy}.root')

            # Use glob to find files matching the source pattern
            source_files = glob.glob(source_pattern)

            if not source_files:
                print("No files found matching the pattern.")
            else:
                # Move each matched file to the destination
                for source_file in source_files:
                    try:
                        print(f"Moving {source_file} to {destination}")
                        shutil.move(source_file, destination)
                        print("File moved successfully.")
                    except Exception as e:
                        print(f"Error moving file {source_file}: {e}")
                        
            # Define the files to remove
            files_to_remove = [os.path.join(toy_output_dir, 'Combine', self.outdir, self.fitFolderName, 'postFit', f'SplusBModels_{cat}', 'toys', f'gen_{toy}.root'), os.path.join(toy_output_dir, 'Combine', self.outdir, self.fitFolderName, 'postFit', f'SplusBModels_{cat}', 'toys', f'fit_{toy}.root')]

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
                file_size = os.path.getsize(os.path.join(toy_output_dir, 'Combine', self.outdir, self.fitFolderName, 'postFit', f'SplusBModels_{cat}', 'toys', f'toy_{toy}.root'))  # Get the file size in bytes
                if file_size > 1000:
                    with open(os.path.join(toy_output_dir, 'Combine', self.outdir, self.fitFolderName, 'postFit', f'SplusBModels_{cat}', 'toys', 'filechecker', f'toy_{toy}_ok.txt'), 'w') as f:
                        pass
            except OSError:
                # Handle the case where the file does not exist or is inaccessible
                print(f"Error creating file. Probably I/O error.")
                return False

        else:

            execute_command([f'mkdir -p {output_dir}/Combine/{self.outdir}/{self.fitFolderName}/preFit/SplusBModels_{cat}/toys/filechecker'], shell=True)
            os.chdir(os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'preFit', f'SplusBModels_{cat}', 'toys'))

            arguments = [
                "combine",
                "-M", "GenerateOnly",
                "-d", self.datacard_path,
                "-m", f"{HIGGS_MASS}",
                "--saveWorkspace",
                "--toysFrequentist",
                "--bypassFrequentistFit",
                "-t", "1",
                "-s", "-1",
                "-n", f"_{toy}_gen_step",
            ]
            arguments.append("--setParameters")
            if self.variable in ['', 'tuto']:
                arguments.append('r=1')
            elif self.variable == 'MH':
                arguments.append(f'r=1,MH={HIGGS_MASS}')
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
            toy_output_dir = output_dir
                
            source_pattern = os.path.join(toy_output_dir, 'Combine', self.outdir, self.fitFolderName, 'preFit', f'SplusBModels_{cat}', 'toys', f'higgsCombine_{toy}_gen_step*.root')
            destination = os.path.join(toy_output_dir, 'Combine', self.outdir, self.fitFolderName, 'preFit', f'SplusBModels_{cat}', 'toys', f'gen_{toy}.root')

            # Use glob to find files matching the source pattern
            source_files = glob.glob(source_pattern)

            if not source_files:
                print("No files found matching the pattern.")
            else:
                # Move each matched file to the destination
                for source_file in source_files:
                    try:
                        print(f"Moving {source_file} to {destination}")
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
            if self.variable in ['', 'tuto']:
                arguments.append('r=1')
            elif self.variable == 'MH':
                arguments.append(f'r=1,MH={HIGGS_MASS}')
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
            source_pattern = os.path.join(toy_output_dir, 'Combine', self.outdir, self.fitFolderName, 'preFit', f'SplusBModels_{cat}', 'toys', f'higgsCombine_{toy}_fit_step*.root')
            destination = os.path.join(toy_output_dir, 'Combine', self.outdir, self.fitFolderName, 'preFit', f'SplusBModels_{cat}', 'toys', f'fit_{toy}.root')

            # Use glob to find files matching the source pattern
            source_files = glob.glob(source_pattern)

            if not source_files:
                print("No files found matching the pattern.")
            else:
                # Move each matched file to the destination
                for source_file in source_files:
                    try:
                        print(f"Moving {source_file} to {destination}")
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
            source_pattern = os.path.join(toy_output_dir, 'Combine', self.outdir, self.fitFolderName, 'preFit', f'SplusBModels_{cat}', 'toys', f'higgsCombine_{toy}_throw_step*.root')
            destination = os.path.join(toy_output_dir, 'Combine', self.outdir, self.fitFolderName, 'preFit', f'SplusBModels_{cat}', 'toys', f'toy_{toy}.root')

            # Use glob to find files matching the source pattern
            source_files = glob.glob(source_pattern)

            if not source_files:
                print("No files found matching the pattern.")
            else:
                # Move each matched file to the destination
                for source_file in source_files:
                    try:
                        print(f"Moving {source_file} to {destination}")
                        shutil.move(source_file, destination)
                        print("File moved successfully.")
                    except Exception as e:
                        print(f"Error moving file {source_file}: {e}")
                        
            # Define the files to remove
            files_to_remove = [os.path.join(toy_output_dir, 'Combine', self.outdir, self.fitFolderName, 'preFit', f'SplusBModels_{cat}', 'toys', f'gen_{toy}.root'), os.path.join(toy_output_dir, 'Combine', self.outdir, self.fitFolderName, 'preFit', f'SplusBModels_{cat}', 'toys', f'fit_{toy}.root')]

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
                file_size = os.path.getsize(os.path.join(toy_output_dir, 'Combine', self.outdir, self.fitFolderName, 'preFit', f'SplusBModels_{cat}', 'toys', f'toy_{toy}.root'))  # Get the file size in bytes
                if file_size > 1000:
                    with open(os.path.join(toy_output_dir, 'Combine', self.outdir, self.fitFolderName, 'preFit', f'SplusBModels_{cat}', 'toys', 'filechecker', f'toy_{toy}_ok.txt'), 'w') as f:
                        pass
            except OSError:
                # Handle the case where the file does not exist or is inaccessible
                print(f"Error creating file. Probably I/O error.")
                return False
        
        os.chdir(cwd)
        
class MggDistribution(CombineMultiYearTask): #(law.Task): #(CombineTask, HTCondorWorkflow, law.LocalWorkflow):

    is_postfit = luigi.BoolParameter(default=False, description="Flag that signifies if toys are created for postfit mass distributions.")

    # def requires(self):
    def _requires_single(self):
        
        tasks = {}
        
        config = self.get_input_config()
        output_dir = self.get_output_dir()
        
        mggConfig = config['combine_mggToys']
        
        if config['combine_mggToys']['doBands']:
            if self.is_postfit:
                tasks["MggToyGeneration"] = MggToyGeneration.req(
                    self,
                    output_dir=output_dir, 
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
                    workflow=mggConfig["execution"], 
                    slurm_partition=mggConfig.get('batchPartition', None), 
                    slurm_memory=mggConfig.get('batchMemory', None), 
                    slurm_max_runtime=mggConfig.get('batchMaxRuntime', None), 
                    htcondor_partition=mggConfig.get('batchPartition', None), 
                    htcondor_memory=mggConfig.get('batchMemory', None), 
                    htcondor_max_runtime=mggConfig.get('batchMaxRuntime', None))
        else:
            if self.is_postfit:
                fitConfig = config["combine_fit"]
                # tasks["CreateUnblindedFit"] = CreateUnblindedFit.req(
                #     self,
                #     output_dir=output_dir, 
                #     version=f"{self.variable if self.variable != '' else 'r'}_prefit", 
                #     workflow=fitConfig["execution"], 
                #     slurm_partition=fitConfig['batchPartition'], 
                #     slurm_memory=fitConfig['batchMemory'], 
                #     slurm_max_runtime=fitConfig['batchMaxRuntime'], 
                #     htcondor_partition=fitConfig['batchPartition'], 
                #     htcondor_memory=fitConfig['batchMemory'], 
                #     htcondor_max_runtime=fitConfig['batchMaxRuntime'])
            else:
                tasks["RunT2WS"] = RunText2Workspace.req(
                    self,
                    output_dir=output_dir, 
                    )
        
        return tasks


    def output(self):

        cat = self.variable
        
        output_dir = self.get_output_dir()

        if self.variable == 'MH':
            years = yearMap[self.year]
            if len(years) > 1:
                reco_cats_with_bmw = ['all']
            else:
                reco_cats_with_bmw = self.get_cats(return_list=True)

        elif self.variable == '':
            reco_cats_with_bmw = ['cat0', 'cat1', 'cat2']
            if "_" in self.year:
                cats = []
                for y in self.year.split("_"):
                    y2 = y[-2:]
                    for c in reco_cats_with_bmw:
                        cats.append(f"Y{y2}_{c}")
                reco_cats_with_bmw = cats

        else:
            reco_cats_with_bmw = [element for element in combineVariableDict(self.variable, self.year)['catsStrWithBMW'] if "_".join(cat.split("_")[2:]) in element]
            if "_" in self.year:
                cats = []
                for y in self.year.split("_"):
                    y2 = y[-2:]
                    for c in reco_cats_with_bmw:
                        cats.append(f"Y{y2}_{c}")
                reco_cats_with_bmw = cats
        
        output = []
        if self.is_postfit:
            output += [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'postFit', 'jsons')]
            output += [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'postFit', 'jsons', f'catsWeights_sospb_{cat}_CMS_hgg_mass.json')]
            
            output += [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'postFit', f'SplusBModels_{cat}', f'{cat}_{catWithBMW}_CMS_hgg_mass.pdf') for catWithBMW in reco_cats_with_bmw]
            output += [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'postFit', f'SplusBModels_{cat}', f'{cat}_{catWithBMW}_CMS_hgg_mass.png') for catWithBMW in reco_cats_with_bmw]
            output += [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'postFit', f'SplusBModels_{cat}', f'{cat}_all_CMS_hgg_mass.pdf')]
            output += [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'postFit', f'SplusBModels_{cat}', f'{cat}_all_CMS_hgg_mass.png')]
            output += [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'postFit', f'SplusBModels_{cat}', f'{cat}_wall_CMS_hgg_mass.pdf')]
            output += [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'postFit', f'SplusBModels_{cat}', f'{cat}_wall_CMS_hgg_mass.png')]
        else:
            output += [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'preFit', 'jsons')]
            output += [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'preFit', 'jsons', f'catsWeights_sospb_{cat}_CMS_hgg_mass.json')]
            
            output += [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'preFit', f'SplusBModels_{cat}', f'{cat}_{catWithBMW}_CMS_hgg_mass.pdf') for catWithBMW in reco_cats_with_bmw]
            output += [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'preFit', f'SplusBModels_{cat}', f'{cat}_{catWithBMW}_CMS_hgg_mass.png') for catWithBMW in reco_cats_with_bmw]
            output += [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'preFit', f'SplusBModels_{cat}', f'{cat}_all_CMS_hgg_mass.pdf')]
            output += [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'preFit', f'SplusBModels_{cat}', f'{cat}_all_CMS_hgg_mass.png')]
            output += [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'preFit', f'SplusBModels_{cat}', f'{cat}_wall_CMS_hgg_mass.pdf')]
            output += [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'preFit', f'SplusBModels_{cat}', f'{cat}_wall_CMS_hgg_mass.png')]
                
        
        outputFileTargets = []
                
        for _, current_output_path in enumerate(output):
            outputFileTargets.append(law.LocalFileTarget(current_output_path))
            
        # print(outputFileTargets)

        return {self.year: outputFileTargets}

    def run(self):
        cat = self.variable

        config = self.get_input_config()
        output_dir = self.get_output_dir()
        main_dir = os.getcwd()
        stage = 'postFit' if self.is_postfit else 'preFit'
        stage_dir = os.path.join(
            output_dir, 'Combine', self.outdir, self.fitFolderName, stage
        )
        plot_dir = os.path.join(stage_dir, f'SplusBModels_{cat}')
        execute_command([f'mkdir -p {plot_dir}'], shell=True)

        skipIndivCat = False
        if self.variable == '':
            reco_cats_with_bmw = ['cat0', 'cat1', 'cat2']
        elif self.variable == 'tuto' and not self.is_postfit:
            reco_cats_with_bmw = [
                'EBEB_highR9highR9', 'EBEB_highR9lowR9', 'EBEB_lowR9highR9',
                'EBEE_highR9highR9', 'EBEE_highR9lowR9', 'EBEE_lowR9highR9',
                'EEEB_highR9highR9', 'EEEB_highR9lowR9', 'EEEB_lowR9highR9',
                'EEEE_incl',
            ]
        elif self.variable == 'MH':
            if len(yearMap[self.year]) > 1:
                skipIndivCat = True
                reco_cats_with_bmw = ['all']
            else:
                reco_cats_with_bmw = self.get_cats().split(',')
        else:
            cat_suffix = '_'.join(cat.split('_')[2:])
            reco_cats_with_bmw = [
                reco_cat
                for reco_cat in combineVariableDict(
                    self.variable, self.year
                )['catsStrWithBMW']
                if cat_suffix in reco_cat
            ]

        if self.variable not in ('MH', 'tuto') and '_' in self.year:
            reco_cats_with_bmw = [
                f"Y{year[-2:]}_{reco_cat}"
                for year in self.year.split('_')
                for reco_cat in reco_cats_with_bmw
            ]

        if self.is_postfit:
            input_workspace = os.path.join(
                plot_dir,
                f'higgsCombine_bestfit_syst_obs_{cat}.MultiDimFit.mH{HIGGS_MASS}.root',
            )
        else:
            datacard_name = (
                f'Datacard_{self.year}.root' if self.variable == ''
                else f'Datacard_{self.variable}_{self.year}.root'
            )
            input_workspace = os.path.join(
                output_dir, 'Combine', self.outdir, datacard_name
            )

        arguments = [
            'python3',
            os.path.join(
                os.environ['ANALYSIS_PATH'], 'Plots', 'makeSplusBModelPlot.py'
            ),
            '--inputWSFile', input_workspace,
            '--cats', ','.join(reco_cats_with_bmw),
            '--lumiLabel', get_lumi_label(self.year),
            '--isPreliminary',
            '--doZeroes',
            '--translateCats', os.path.join(
                os.environ['ANALYSIS_PATH'], 'Plots', 'cats.json'
            ),
            '--doSumCategories',
            '--doCatWeights',
            '--saveWeights',
            '--ext', f'_{cat}',
            '--POI', cat,
        ]

        if self.is_postfit:
            arguments += [
                '--loadSnapshot', config['combine_mggToys']['loadSnapshot'],
                '--unblind',
                '--pseudoToy',
            ]
        else:
            arguments += [
                '--blindingRegion', '115,135',
                '--mass', f'{HIGGS_MASS}',
                '--showPOIs',
            ]

        if skipIndivCat:
            arguments.append('--skipIndividualCatPlots')
        if config['combine_mggToys']['doBands']:
            arguments += ['--doBands', '--doToyVeto', '--saveToyYields']

        print(' '.join(arguments))
        try:
            os.chdir(stage_dir)
            result = subprocess.run(
                arguments, check=True, text=True, capture_output=True
            )
            print("Script output:", result.stdout)
            print("Script executed successfully.")
        except subprocess.CalledProcessError as exc:
            print("Error executing script:", exc.stderr)
            raise
        finally:
            os.chdir(main_dir)
        
        
class PValueCalculation(CombineTask, law.LocalWorkflow, HTCondorWorkflow, SlurmWorkflow): #(law.Task): #(CombineTask, HTCondorWorkflow, law.LocalWorkflow):

    # def requires(self):
    def workflow_requires(self):
        workflow_reqs = super().workflow_requires()

        tasks = {}

        if workflow_reqs:
            tasks.update(workflow_reqs)
        
        config = self.get_input_config()
        output_dir = self.get_output_dir()
        
        fitConfig = config["combine_fit"]
        tasks["CreateLikelihoodFit"] = CreateLikelihoodFitWrapper.req(
            self,
            output_dir=output_dir,
            years=self.year,
            is_postfit=True,
        )
        
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
            output = []
        else:
            output = [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'dataFit', f'higgsCombine.pvalue.MultiDimFit.mH{HIGGS_MASS}.root')]
            output += [os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, f'pvalue.txt')]

        outputFileTargets = []
                
        for _, current_output_path in enumerate(output):
            outputFileTargets.append(law.LocalFileTarget(current_output_path))
            
        # print(outputFileTargets)

        return outputFileTargets
    
    def run(self):

        #Load central config file
        config = self.get_input_config()
        output_dir = self.get_output_dir()  
            
        cwd = os.getcwd()
        execute_command([f'mkdir -p {output_dir}/Combine/{self.outdir}/{self.fitFolderName}/dataFit'], shell=True)
        os.chdir(os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'dataFit'))
        temp_output_dir = output_dir
                    
        # Define the file to check
        pvalue_file = os.path.join(temp_output_dir, 'Combine', self.outdir, self.fitFolderName, 'dataFit', f'higgsCombine.pvalue.MultiDimFit.mH{HIGGS_MASS}.root')
        # Check if the file exists
        if not os.path.isfile(pvalue_file):
            print("The pvalue file does not exist in the current directory. Creating it...")
            # Iterate over the parameters
            command = [
                "combine",
                "-M", "MultiDimFit",
                os.path.join(output_dir, 'Combine', self.outdir, self.fitFolderName, 'dataFit', f"higgsCombineDataPostFitScanFit_{combineVariableDict(self.variable, self.year)['paramStrNoOne'][0]}.MultiDimFit.mH{HIGGS_MASS}.root"),
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

        # Change directory to the self.fitFolderName
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

        os.chdir(cwd)
