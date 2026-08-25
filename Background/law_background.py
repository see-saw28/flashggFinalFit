import law
import os
import subprocess
import glob
import yaml
import errno
import subprocess
import shutil
import re

from commonTools import *
from commonObjects import *

from Trees2WS.law_trees2ws_data import *

from framework import Task, MultiYearTask
from framework import HTCondorWorkflow, SlurmWorkflow

# Function to safely create a directory
def safe_mkdir(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def execute_command(command, return_output=False, shell=False):
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True, shell=shell, env=os.environ)
        print("Script output:", result.stdout)
        print("Script executed successfully.")
        if return_output:
            return (result.stdout).split("\n")[0]
    except subprocess.CalledProcessError as e:
        print("Error executing script:", e.stderr)
        raise

def get_lumi_label(year):
    if year in lumiMap:
        lumi = lumiMap[year]
    else:
        lumi = sum(lumiMap[y] for y in re.split(r"[,_]", str(year)) if y in lumiMap)

    sqrts = sqrtMap.get(year, 13.6)
    return f"{lumi:.1f} fb^{{-1}} ({sqrts:g} TeV)"

class BackgroundCategory(Task, HTCondorWorkflow, SlurmWorkflow, law.LocalWorkflow):#(law.Task): #(Task, HTCondorWorkflow, law.LocalWorkflow):
    input_path = law.Parameter(description="Path to the alldata input ROOT file")
    output_dir = law.Parameter(description="Path to the output directory")
    ext = law.Parameter(default="earlyAnalysis", description="Extension to be used for output folder naming")
    year = law.Parameter(default='2022', description="Year")
    cats = law.Parameter(description="List of categories separated by a comma.")
    cat_offset = law.Parameter(description="Category offset")
    variable = law.Parameter(default="", description="Variable to be used")

    batch_flavor = law.Parameter(default="htcondor", description="Batch system to use")

    def workflow_requires(self):
        workflow_reqs = super().workflow_requires()

        tasks = {}

        if workflow_reqs:
            tasks.update(workflow_reqs)
        
        config = self.get_input_config()
        output_dir = self.get_output_dir()
        
        config = config["backgroundScriptCfg"]
        
        tasks["Trees2WSData"] = Trees2WSData.req(
            self,
            output_dir=output_dir,
            variable=self.variable,
            year=self.year,
            workflow=config['execution'],
            batch_flavor=self.batch_flavor,
            slurm_partition=config['batchPartition'],
            slurm_memory=config['batchMemory'],
            slurm_max_runtime=config['batchMaxRuntime'],
            htcondor_partition=config['batchPartition'],
            htcondor_memory=config['batchMemory'],
            htcondor_max_runtime=config['batchMaxRuntime']
        )
        
        return tasks
    
    def create_branch_map(self):
        # map branch indexes to ascii numbers from 97 to 122 ("a" to "z")
        nCats = len(self.cats.split(","))
              
        cat_list = [
            (self.cats.split(",")[categoryIndex], str(int(self.cat_offset)+categoryIndex))
            for categoryIndex in range(nCats)
        ]        
        branch_map = {i: cat_catOffset for i, cat_catOffset in enumerate(cat_list)}
        return branch_map

    def output(self):
        cat, cat_offset = self.branch_data
        outdir_ext = os.path.join(self.output_dir, 'Background', f'outdir_{self.ext}')

        
        outputFileTargets = []
        
        bkg_plots = glob.glob(os.path.join(outdir_ext, f'bkgfTest-Data/*_cat{cat_offset}.png'))
        bkg_plots += glob.glob(os.path.join(outdir_ext, f'bkgfTest-Data/*_cat{cat_offset}.pdf'))
        bkg_plots += glob.glob(os.path.join(outdir_ext, f'bkgfTest-Data/*_cat{cat_offset}.pdf_gofTest.pdf'))
    
        output_paths = [os.path.join(outdir_ext, f'CMS-HGG_multipdf_{cat}.root'), os.path.join(outdir_ext, f'bkgfTest-Data/multipdf_{cat}.pdf'), os.path.join(outdir_ext, f'bkgfTest-Data/multipdf_{cat}.png')]
        
        output_paths += bkg_plots

   
        for _, current_output_path in enumerate(output_paths):
            outputFileTargets.append(law.LocalFileTarget(current_output_path))

        return outputFileTargets

    def run(self):
        cat, cat_offset = self.branch_data
        input_path = self.input_path
        
        if self.batch_flavor == "slurm/psi":
            # Have to use /scratch/batch_username/ for slurm/psi
            os.environ["TARGET_PATH"] = f"/scratch/{os.environ['USER']}/{os.environ['SLURM_JOB_ID']}"
            temp_output_dir = os.environ["TARGET_PATH"]
            execute_command([f'xrdfs root://t3dcachedb03.psi.ch:1094/ mkdir -p {self.output_dir}/Background'], shell=True)
            execute_command([f'xrdfs root://t3dcachedb03.psi.ch:1094/ mkdir -p {self.output_dir}/Background/outdir_{self.ext}'], shell=True)
            safe_mkdir(temp_output_dir)
        else:
            safe_mkdir(self.output_dir)
            safe_mkdir(os.path.join(self.output_dir, "Background"))
            safe_mkdir(os.path.join(self.output_dir, "Background", f"outdir_{self.ext}"))
            temp_output_dir = os.path.join(self.output_dir, "Background")
        
        if temp_output_dir[-1] != "/":
            temp_output_dir += "/"

        script_path = os.path.join(os.environ["ANALYSIS_PATH"], "Background/runBackgroundScripts.sh")
        arguments = [
            "-i", input_path,
            "-p", "none",
            "-f", cat,
            "--outputFolder", f"{temp_output_dir}",
            "--ext", f'{self.ext}',
            "--catOffset", cat_offset,
            "--intLumi", f"{lumiMap[self.year]}",
            "--lumiLabel", get_lumi_label(self.year),
            "--year", f"{self.year}",
            "--batch", "local",
            "--queue", "microcentury",
            "--sigFile", "none",
            "--isData",
            "--fTestOnly"
        ]
        command = [script_path] + arguments
        # print("Output:", command)
        
        # Move to background folder
        original_dir = os.getcwd()
        os.chdir(os.path.join(os.environ["ANALYSIS_PATH"], "Background"))
        try:
            result = subprocess.run(command, check=True, text=True, capture_output=True)
            print("Script output:", result.stdout)
            print("Script executed successfully.")
        except subprocess.CalledProcessError as e:
            print("Error executing script:", e.stderr)
            raise
        finally:
            os.chdir(original_dir)

        if self.batch_flavor == "slurm/psi":
            bkg_folder = f"outdir_{self.ext}"
            execute_command([f"ls -al {temp_output_dir}/*"], shell=True)
            if "/work" in self.output_dir:
                slurm_copy_command = [
                    'cp', '-rf',
                    f'{temp_output_dir}/{bkg_folder}',
                     f'{self.output_dir}/Background/'
                ]
            else:
                # Copying output files to final destination on the /pnfs.
                slurm_copy_command = [
                    'xrdcp', '-rf',
                    f'{temp_output_dir}/{bkg_folder}',
                    'root://t3dcachedb03.psi.ch:1094//'+ f'{self.output_dir}/Background/'
                ]
            execute_command(slurm_copy_command)
            # Cleaning up scratch space.
            shutil.rmtree(temp_output_dir)
        

class Background(MultiYearTask):#(law.Task): #(Task, HTCondorWorkflow, law.LocalWorkflow):

    def _requires_single(self):
        config = self.get_input_config()
        output_dir = self.get_output_dir()
        background_config = config["backgroundScriptCfg"]

        input_path = config['inputFiles']['Trees2WSData']
        if background_config['cats'] == 'auto':
            background_config['cats'] = extractListOfCatsFromHiggsDNAAllData(input_path)

        tasks = []

        if self.variable == '':
            all_data_input_path = os.path.join(
                output_dir,
                "Tree2WSData",
                f"input_output_data_{self.year}/ws/allData.root"
            )
        else:
            all_data_input_path = os.path.join(
                output_dir,
                "Tree2WSData",
                f"input_output_data_{self.variable}_{self.year}/ws/allData.root"
            )

        tasks.append(BackgroundCategory.req(
            self,
            input_path=all_data_input_path,
            output_dir=output_dir,
            ext=background_config['ext'],
            year=self.year,
            cats=background_config['cats'],
            cat_offset=background_config['catOffset'],
            workflow=background_config['execution'],
            slurm_partition=background_config['batchPartition'],
            slurm_memory=background_config['batchMemory'],
            slurm_max_runtime=background_config['batchMaxRuntime'],
            htcondor_partition=background_config['batchPartition'],
            htcondor_memory=background_config['batchMemory'],
            htcondor_max_runtime=background_config['batchMaxRuntime']))

        return tasks

    def output(self):
        return self.input()

    def run(self):
        return True
        
