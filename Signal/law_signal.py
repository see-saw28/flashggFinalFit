import law
import os, sys
import subprocess
import glob
import yaml
import errno
import time

from commonTools import *
from commonObjects import *
from Trees2WS.law_trees2ws import *

from framework import Task, MultiYearTask
from framework import HTCondorWorkflow, SlurmWorkflow

sys.path.append(os.path.dirname(os.path.abspath(__file__))+ "/tools")

# Function to safely create a directory
def safe_mkdir(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def count_files_in_directory(directory='.'):
    return len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])

def convert_boolean_string(string):
    if (string == "True") or (string == "true") or (string == True):
        return True
    else:
        return False              

class FTestCategory(Task, HTCondorWorkflow, SlurmWorkflow, law.LocalWorkflow): #(Task): #(Task, HTCondorWorkflow, law.LocalWorkflow):
    input_path = law.Parameter(description="Path to the input ROOT files (/ws_signal)")
    output_dir = law.Parameter(description="Path to the output directory")
    ext = law.Parameter(default="earlyAnalysis", description="Extension to be used for output folder naming")
    cats = law.Parameter(description="Category string")
    procs = law.Parameter(description="Processes")
    variable = law.Parameter(default="", description="Variable to be used")
    year = law.Parameter(description="Year")    
    era = law.Parameter(default="", description="Current Era")    
    
    batch_flavor = law.Parameter(default="htcondor", description="Batch system to use")

    htcondor_job_kwargs_submit = {"spool": True}
    
    # def requires(self):
    def workflow_requires(self):
        workflow_reqs = super().workflow_requires()

        tasks = {}

        if workflow_reqs:
            tasks.update(workflow_reqs)
        
        # Load the input configuration
        config = self.get_input_config()
        
        output_dir = self.get_output_dir()
            
        tasks["Trees2WS"] = Trees2WS.req(self, output_dir=output_dir)
        
        return tasks
    
    def create_branch_map(self):
        # map branch indexes to ascii numbers from 97 to 122 ("a" to "z")
        nCats = len(self.cats.split(","))
        
        cat_list = [
            self.cats.split(",")[categoryIndex]
            for categoryIndex in range(nCats)
        ]
        
        branch_map = {i: cat for i, cat in enumerate(cat_list)}
        return branch_map

    def output(self):
        
        cat = self.branch_data

        ftest_output = [os.path.join(self.output_dir, "Signal", f'outdir_{self.ext}/fTest/json/nGauss_{cat}.json')]
                
        outputFileTargets = []
                
        for _, current_output_path in enumerate(ftest_output):
            outputFileTargets.append(law.LocalFileTarget(current_output_path))

        # output_paths.append(law.LocalFileTarget(os.path.join(output_dir, f"outdir_{currentConfig['ext']}/fTest/Plots")))

        return outputFileTargets

    def run(self):
        cat = self.branch_data
        
        sys.path.append(os.path.dirname(os.path.abspath(__file__))+ "/tools")

        if self.batch_flavor == "slurm/psi":
            # Have to use /scratch/batch_username/ for slurm/psi
            if "/work" in self.output_dir:
                execute_command([f'mkdir -p {self.output_dir}/Signal/outdir_{self.ext}/fTest/Plots'], shell=True)
                execute_command([f'mkdir -p {self.output_dir}/Signal/outdir_{self.ext}/fTest/json'], shell=True)
            else:   
                execute_command([f'xrdfs root://t3dcachedb03.psi.ch:1094/ mkdir -p {self.output_dir}/Signal/outdir_{self.ext}/fTest/Plots'], shell=True)
                execute_command([f'xrdfs root://t3dcachedb03.psi.ch:1094/ mkdir -p {self.output_dir}/Signal/outdir_{self.ext}/fTest/json'], shell=True)

            os.environ["TARGET_PATH"] = f"/scratch/{os.environ['USER']}/{os.environ['SLURM_JOB_ID']}"
            execute_command([f'mkdir -p $TARGET_PATH/outdir_{self.ext}/fTest/Plots'], shell=True)
            execute_command([f'mkdir -p $TARGET_PATH/outdir_{self.ext}/fTest/json'], shell=True)
            output_dir = os.environ["TARGET_PATH"]
            # os.chdir(os.path.join(os.environ["TARGET_PATH"], 'Combine', fitFolderName, 'impact'))
        else:
            execute_command([f'mkdir -p {self.output_dir}/Signal/outdir_{self.ext}/fTest/Plots'], shell=True)
            execute_command([f'mkdir -p {self.output_dir}/Signal/outdir_{self.ext}/fTest/json'], shell=True)
            output_dir = os.path.join(self.output_dir, "Signal")
            # os.chdir(os.path.join(self.output_dir, 'Combine', fitFolderName, 'impact'))

        script_path = os.path.join(os.environ["ANALYSIS_PATH"], "Signal/scripts/fTest.py")
        arguments = [
            "python3",
            script_path,
            "--cat", cat,
            "--procs", self.procs,
            "--ext", self.ext,
            "--outputDir", f"{output_dir}",
            "--inputWSDir", f"{self.input_path}",
            "--doPlots"
        ]
        command = arguments
        print(' '.join(command))
        try:
            result = subprocess.run(command, check=True, text=True, capture_output=True)
            print("Script output:", result.stdout)
            print("Script executed successfully.")
        except subprocess.CalledProcessError as e:
            print("Script output:", e.stdout)
            print("Error executing script:", e.stderr)
            raise

        # Copy the files back to pnfs if we are on slurm/psi
        if self.batch_flavor == "slurm/psi":
            # Have to copy over the output to the final directory
            # Don't forget to VOMS!
            if "/work" in self.output_dir:
                slurm_copy_command = [
                    'cp', '-rf',
                    f"{os.environ['TARGET_PATH']}/outdir_{self.ext}/",
                    self.output_dir
                ]
            else:
                slurm_copy_command = [
                    'xrdcp', '-rf',
                    f"{os.environ['TARGET_PATH']}/outdir_{self.ext}/",
                    'root://t3dcachedb03.psi.ch:1094//'+self.output_dir
                ]
            print(slurm_copy_command)
            execute_command(slurm_copy_command)
            # Clean up the temporary directory
            shutil.rmtree(os.environ["TARGET_PATH"])

class FTest(MultiYearTask):
    variable = law.Parameter(default="", description="Variable to be used")
    output_dir = law.Parameter(default="", description="Path to the output directory")
    year = law.Parameter(default='2022', description="Year")
    
    batch_flavor = law.Parameter(default="htcondor", description="Batch system to use")
    
    def _requires_single(self):
        # req() is defined on all tasks and handles the passing of all parameter values that are
        # common between the required task and the instance (self)
        

        # Load the input configuration
        config = self.get_input_config()
        
        output_dir = self.get_output_dir()
            
        # Use allData.root from HiggsDNA to automatically determine categories
        data_input_path = config['inputFiles']['Trees2WSData']  
        inOutSplittingFlag = config['trees2wsCfg']['doInOutSplitting'] or config['trees2wsCfg']['doDiffSplitting'] # We do the in out splitting for the differentials in HIG-23-014
        signal_input_path = glob.glob(config['inputFiles']['Trees2WS']+'/*')
        
        tasks = []
        
        # Loop over a years era
        eras = allErasMap.get(f"{self.year}", [""])

        for currentEra in eras:

            era_suffix = "" if currentEra in ["", "None"] else currentEra

            if self.variable == "":
                input_path = os.path.join(
                    output_dir,
                    "Tree2WS",
                    f"input_output_{self.year}{era_suffix}/ws_signal"
                )
            else:
                input_path = os.path.join(
                    output_dir,
                    "Tree2WS",
                    f"input_output_{self.variable}_{self.year}{era_suffix}/ws_signal"
                )

            if currentEra not in ["", "None"]:
                currentConfig = config[f"signalScriptCfg_{self.year}_{currentEra}"]
            else:
                currentConfig = config[f"signalScriptCfg_{self.year}"]

            # Extract low and high MH values
            mps = []
            for mp in currentConfig['massPoints'].split(","): mps.append(int(mp))
            currentConfig['massLow'], currentConfig['massHigh'] = '%s'%min(mps), '%s'%max(mps)

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # If proc/cat == auto. Extract processes and categories
            if currentConfig['cats'] == "auto":
                currentConfig['cats'] = extractListOfCatsFromHiggsDNAAllData(data_input_path)
            currentConfig['nCats'] = len(currentConfig['cats'].split(","))

            if currentConfig['procs'] == "auto":
                currentConfig['procs'] = extractListOfProcsFromHiggsDNASignal(signal_input_path, self.variable, inOutSplittingFlag)
            currentConfig['nProcs'] = len(currentConfig['procs'].split(","))

            for cat in currentConfig['cats'].split(","):
                tasks.append(FTestCategory.req(
                    self,
                    input_path=input_path, 
                    output_dir=output_dir, 
                    ext=currentConfig['ext'], 
                    cats=cat, 
                    procs=currentConfig['procs'], 
                    workflow=currentConfig['execution'], 
                    era=currentEra, 
                    slurm_partition=currentConfig['batchPartition'], 
                    slurm_memory=currentConfig['batchMemory'], 
                    slurm_max_runtime=currentConfig['batchMaxRuntime'], 
                    htcondor_partition=currentConfig['batchPartition'], 
                    htcondor_memory=currentConfig['batchMemory'], 
                    htcondor_max_runtime=currentConfig['batchMaxRuntime']))

        return tasks

    def output(self):
        return self.input()
                
    
    def run(self):
        
        return True
    
    
class CalcPhotonSystCategory(Task, HTCondorWorkflow, SlurmWorkflow, law.LocalWorkflow):#(Task): #(Task, HTCondorWorkflow, law.LocalWorkflow):
    input_path = law.Parameter(description="Path to the input ROOT files (/ws_signal)")
    output_dir = law.Parameter(description="Path to the output directory")
    ext = law.Parameter(default="earlyAnalysis", description="Extension to be used for output folder naming")
    cats = law.Parameter(description="Category string")
    procs = law.Parameter(description="Processes")
    scales = law.Parameter(description="Scales")
    scalesCorr = law.Parameter(default="", description="Scale corrections")
    scalesGlobal = law.Parameter(default="", description="Global scales")
    smears = law.Parameter(description="Smearings")
    variable = law.Parameter(default="", description="Variable to be used")
    year = law.Parameter(description="Year")    
    era = law.Parameter(description="era")    
    
    batch_flavor = law.Parameter(default="htcondor", description="Batch system to use")
    
    htcondor_job_kwargs_submit = {"spool": True}

    # def requires(self):
    def workflow_requires(self):
        workflow_reqs = super().workflow_requires()

        tasks = {}

        if workflow_reqs:
            tasks.update(workflow_reqs)
        
        # Load the input configuration
        config = self.get_input_config()
        
        output_dir = self.get_output_dir()
            
        tasks["Trees2WS"] = Trees2WS.req(self, output_dir=output_dir)
        
        return tasks
    
    def create_branch_map(self):
        # map branch indexes to ascii numbers from 97 to 122 ("a" to "z")
        nCats = len(self.cats.split(","))
        
        cat_list = [
            self.cats.split(",")[categoryIndex]
            for categoryIndex in range(nCats)
        ]
        
        branch_map = {i: cat for i, cat in enumerate(cat_list)}
        return branch_map

    def output(self):
        
        cat = self.branch_data
        
        ftest_output = [os.path.join(self.output_dir, f'Signal/outdir_{self.ext}/calcPhotonSyst/pkl/{cat}.pkl')]
                
        outputFileTargets = []
        
                
        for _, current_output_path in enumerate(ftest_output):
            outputFileTargets.append(law.LocalFileTarget(current_output_path))

        return outputFileTargets

    def run(self):
        
        cat = self.branch_data

        sys.path.append(os.path.dirname(os.path.abspath(__file__))+ "/tools")

        if self.batch_flavor == "slurm/psi":
            # Have to use /scratch/batch_username/ for slurm/psi
            if "/work" in self.output_dir:
                execute_command([f'mkdir -p {self.output_dir}/outdir_{self.ext}/Signal/calcPhotonSyst/pkl'], shell=True)
            else:   
                execute_command([f'xrdfs root://t3dcachedb03.psi.ch:1094/ mkdir -p {self.output_dir}/Signal/calcPhotonSyst/pkl'], shell=True)

            os.environ["TARGET_PATH"] = f"/scratch/{os.environ['USER']}/{os.environ['SLURM_JOB_ID']}"
            execute_command([f'mkdir -p $TARGET_PATH/outdir_{self.ext}/Signal/calcPhotonSyst/pkl'], shell=True)
            output_dir = os.environ["TARGET_PATH"]
            # os.chdir(os.path.join(os.environ["TARGET_PATH"], 'Combine', fitFolderName, 'impact'))
        else:
            execute_command([f'mkdir -p {self.output_dir}/Signal/outdir_{self.ext}/calcPhotonSyst/pkl'], shell=True)
            output_dir = os.path.join(self.output_dir, "Signal")
            print(f"Output directory: {output_dir}")
            # os.chdir(os.path.join(self.output_dir, 'Combine', fitFolderName, 'impact'))

        script_path = os.path.join(os.environ["ANALYSIS_PATH"], "Signal/scripts/calcPhotonSyst.py")
        arguments = [
            "python3",
            script_path,
            "--cat", cat,
            "--procs", self.procs,
            "--ext", self.ext,
            "--outputDir", f"{output_dir}",
            "--inputWSDir", f"{self.input_path}",
            "--scales", f"{self.scales}",
            "--smears", f"{self.smears}",
            "--doPlots"
        ]
        if self.scalesCorr != "":
            arguments.append("--scalesCorr")
            arguments.append("%s"%self.scalesCorr)
        if self.scalesCorr != "":
            arguments.append("--scalesGlobal")
            arguments.append("%s"%self.scalesGlobal)
        command = arguments
        print(' '.join(command))
        try:
            result = subprocess.run(command, check=True, text=True, capture_output=True)
            print("Script output:", result.stdout)
            print("Script executed successfully.")
        except subprocess.CalledProcessError as e:
            print("Script output:", e.stdout)
            print("Error executing script:", e.stderr)
            raise

        # Copy the files back to pnfs if we are on slurm/psi
        if self.batch_flavor == "slurm/psi":
            # Have to copy over the output to the final directory
            # Don't forget to VOMS!
            if "/work" in self.output_dir:
                slurm_copy_command = [
                    'cp', '-rf',
                    f"{os.environ['TARGET_PATH']}/outdir_{self.ext}/",
                    self.output_dir
                ]
            else:
                slurm_copy_command = [
                    'xrdcp', '-rf',
                    f"{os.environ['TARGET_PATH']}/outdir_{self.ext}/",
                    'root://t3dcachedb03.psi.ch:1094//'+self.output_dir
                ]
            print(slurm_copy_command)
            execute_command(slurm_copy_command)
            # Clean up the temporary directory
            shutil.rmtree(os.environ["TARGET_PATH"])

class CalcPhotonSyst(MultiYearTask):
    variable = law.Parameter(default="", description="Variable to be used")
    output_dir = law.Parameter(default="", description="Path to the output directory")
    year = law.Parameter(default='2022', description="Year")

    batch_flavor = law.Parameter(default="htcondor", description="Batch system to use")

    def _requires_single(self):
        # req() is defined on all tasks and handles the passing of all parameter values that are
        # common between the required task and the instance (self)
        

        # Load the input configuration
        config = self.get_input_config()
        
        output_dir = self.get_output_dir()
        
        # Use allData.root from HiggsDNA to automatically determine categories
        data_input_path = config['inputFiles']['Trees2WSData']  
        inOutSplittingFlag = config['trees2wsCfg']['doInOutSplitting'] or config['trees2wsCfg']['doDiffSplitting']
        
        signal_input_path = glob.glob(config['inputFiles']['Trees2WS']+'/*')
        
        tasks = []
        
        # Loop over a years era
        eras = allErasMap.get(f"{self.year}", [""])
        
        for currentEra in eras:
            
            era_suffix = "" if currentEra in ["", "None"] else currentEra
            
            if self.variable == '':
                input_path = os.path.join(output_dir, "Tree2WS", f"input_output_{self.year}{era_suffix}/ws_signal")
            else:
                input_path = os.path.join(output_dir, "Tree2WS", f"input_output_{self.variable}_{self.year}{era_suffix}/ws_signal")


            if currentEra not in ["", "None"]:
                currentConfig = config[f"signalScriptCfg_{self.year}_{era_suffix}"]
            else:
                currentConfig = config[f"signalScriptCfg_{self.year}"]
                
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # If proc/cat == auto. Extract processes and categories
            if currentConfig['cats'] == "auto":
                currentConfig['cats'] = extractListOfCatsFromHiggsDNAAllData(data_input_path)
            currentConfig['nCats'] = len(currentConfig['cats'].split(","))
            
            if currentConfig['procs'] == "auto":
                currentConfig['procs'] = extractListOfProcsFromHiggsDNASignal(signal_input_path, self.variable, inOutSplittingFlag)
            currentConfig['nProcs'] = len(currentConfig['procs'].split(","))
            
            # Extract low and high MH values
            mps = []
            for mp in currentConfig['massPoints'].split(","): mps.append(int(mp))
            currentConfig['massLow'], currentConfig['massHigh'] = '%s'%min(mps), '%s'%max(mps)
                    
            tasks.append(CalcPhotonSystCategory.req(
                self,
                input_path=input_path, 
                output_dir=output_dir, 
                ext=currentConfig['ext'], 
                cats=currentConfig['cats'], 
                procs=currentConfig['procs'], 
                scales=currentConfig['scales'], 
                scalesCorr=currentConfig['scalesCorr'], 
                scalesGlobal=currentConfig['scalesGlobal'], 
                smears=currentConfig['smears'], 
                era=currentEra,
                # version=f"v{i}", 
                workflow=currentConfig['execution'], 
                slurm_partition=currentConfig['batchPartition'], 
                slurm_memory=currentConfig['batchMemory'], 
                slurm_max_runtime=currentConfig['batchMaxRuntime'], 
                htcondor_partition=currentConfig['batchPartition'], 
                htcondor_memory=currentConfig['batchMemory'], 
                htcondor_max_runtime=currentConfig['batchMaxRuntime']))

        return tasks
    
    def output(self):
        return self.input()
                
    
    def run(self):
        return True
    




class SignalFitCategoryProcess(Task, HTCondorWorkflow, SlurmWorkflow, law.LocalWorkflow):#(Task): #(Task, HTCondorWorkflow, law.LocalWorkflow):
    input_path = law.Parameter(description="Path to the input ROOT files (/ws_signal)")
    output_dir = law.Parameter(description="Path to the output directory")
    ext = law.Parameter(default="earlyAnalysis", description="Extension to be used for output folder naming")
    cats = law.Parameter(description="Category list")
    procs = law.Parameter(description="Process list")
    scales = law.Parameter(description="Scales")
    scalesCorr = law.Parameter(default="",description="Scale corrections")
    scalesGlobal = law.Parameter(default="",description="Global scales")
    smears = law.Parameter(description="Smearings")
    year = law.Parameter(description="Year")    
    era = law.Parameter(description="era")   
    analysisXSBR = law.Parameter(description="XSBR Analysis")
    analysisRM = law.Parameter(description="Replacement Map Analysis")
    replacementThreshold = law.Parameter(description="replacementThreshold")
    massPoints = law.Parameter(description="Mass Points")
    beamspotWidthData = law.Parameter(description="Beamspot width in Data")
    beamspotWidthMC = law.Parameter(description="Beamspot width in MC")
    doPlots = law.Parameter(description="Printing the signal models.")
    
    variable = law.Parameter(default="", description="Variable to be used")
    
    batch_flavor = law.Parameter(default="htcondor", description="Batch system to use")
    
    htcondor_job_kwargs_submit = {"spool": True}
      
    # def requires(self):
    def workflow_requires(self):
        workflow_reqs = super().workflow_requires()

        tasks = {}

        if workflow_reqs:
            tasks.update(workflow_reqs)
        
        year = self.year[:4]
        

        output_dir = self.get_output_dir(truncate=True)
            
        tasks["FTest"] = FTest.req(self, output_dir=output_dir, years=year)
        tasks["CalcPhotonSyst"] = CalcPhotonSyst.req(self, output_dir=output_dir, years=year)

        return tasks
    
    def create_branch_map(self):
        # map branch indexes to ascii numbers from 97 to 122 ("a" to "z")
        nCats = len(self.cats.split(","))
        nProcs = len(self.procs.split(","))
        
        cat_proc_list = [
            (self.cats.split(",")[categoryIndex], self.procs.split(",")[processIndex])
            for categoryIndex in range(nCats)
            for processIndex in range(nProcs)
        ]
        
        branch_map = {i: cat_proc for i, cat_proc in enumerate(cat_proc_list)}
        
        return branch_map

    def output(self):
                
        cat, proc = self.branch_data
        
        signal_output = [os.path.join(self.output_dir, "Signal", f'outdir_{self.ext}/signalFit/output/CMS-HGG_sigfit_{self.ext}_{proc}_{self.year}_{cat}.root')]
        signal_output += glob.glob(os.path.join(self.output_dir, "Signal", f'outdir_{self.ext}/signalFit/Plots/{proc}_{self.year}_{cat}*'))
                
        outputFileTargets = []
            
        for _, current_output_path in enumerate(signal_output):
            outputFileTargets.append(law.LocalFileTarget(current_output_path))

        return outputFileTargets

    def run(self):
        
        cat, proc = self.branch_data
        
        sys.path.append(os.path.dirname(os.path.abspath(__file__))+ "/tools")

        if self.batch_flavor == "slurm/psi":
            # Have to use /scratch/batch_username/ for slurm/psi
            if "/work" in self.output_dir:
                execute_command([f'mkdir -p {self.output_dir}/outdir_{self.ext}/signalFit/output'], shell=True)
                execute_command([f'mkdir -p {self.output_dir}/outdir_{self.ext}/signalFit/Plots'], shell=True)
            else:   
                execute_command([f'xrdfs root://t3dcachedb03.psi.ch:1094/ mkdir -p {self.output_dir}/signalFit/output'], shell=True)
                execute_command([f'xrdfs root://t3dcachedb03.psi.ch:1094/ mkdir -p {self.output_dir}/signalFit/Plots'], shell=True)

            os.environ["TARGET_PATH"] = f"/scratch/{os.environ['USER']}/{os.environ['SLURM_JOB_ID']}"
            execute_command([f'mkdir -p $TARGET_PATH/outdir_{self.ext}/signalFit/output'], shell=True)
            execute_command([f'mkdir -p $TARGET_PATH/outdir_{self.ext}/signalFit/Plots'], shell=True)
            output_dir = os.environ["TARGET_PATH"]
        else:
            execute_command([f'mkdir -p {self.output_dir}/Signal/outdir_{self.ext}/signalFit/output'], shell=True)
            execute_command([f'mkdir -p {self.output_dir}/Signal/outdir_{self.ext}/signalFit/Plots'], shell=True)
            output_dir = os.path.join(self.output_dir, "Signal")

        script_path = os.path.join(os.environ["ANALYSIS_PATH"], "Signal/scripts/signalFit.py")
        arguments = [
            "python3",
            script_path,
            "--cat", cat,
            "--proc", proc,
            "--ext", self.ext,
            "--outputDir", f"{output_dir}",
            "--inputWSDir", f"{self.input_path}",
            "--year", f"{self.year}",
            "--scales", f"{self.scales}",
            "--smears", f"{self.smears}",
            "--analysisXSBR", f"{self.analysisXSBR}",
            "--analysisRM", f"{self.analysisRM}",
            "--massPoints", f"{self.massPoints}",
            "--replacementThreshold",f"{self.replacementThreshold}",
            "--beamspotWidthData", f"{self.beamspotWidthData}",
            "--beamspotWidthMC", f"{self.beamspotWidthMC}"
        ]
        if convert_boolean_string(self.doPlots):
            arguments += ["--doPlots"]
        if self.scalesCorr != "":
            arguments += ["--scalesCorr"]
            arguments += ["%s"%self.scalesCorr]
        if self.scalesGlobal != "":
            arguments += ["--scalesGlobal"]
            arguments += ["%s"%self.scalesGlobal]
        if (self.batch_flavor == "slurm/psi"):
            arguments += ["--ingredientsDir"]
            arguments += ["%s"%self.output_dir]
        command = arguments
        print(' '.join(command))
        try:
            result = subprocess.run(command, check=True, text=True, capture_output=True)
            print("Script output:", result.stdout)
            print("Script executed successfully.")
        except subprocess.CalledProcessError as e:
            print("Script output:", e.stdout)
            print("Error executing script:", e.stderr)
            raise

        # Copy the files back to pnfs if we are on slurm/psi
        if self.batch_flavor == "slurm/psi":
            # Have to copy over the output to the final directory
            # Don't forget to VOMS!
            if "/work" in self.output_dir:
                slurm_copy_command = [
                    'cp', '-rf',
                    f"{os.environ['TARGET_PATH']}/outdir_{self.ext}/",
                    self.output_dir
                ]
            else:
                slurm_copy_command = [
                    'xrdcp', '-rf',
                    f"{os.environ['TARGET_PATH']}/outdir_{self.ext}/",
                    'root://t3dcachedb03.psi.ch:1094//'+self.output_dir
                ]
            print(slurm_copy_command)
            execute_command(slurm_copy_command)
            # Clean up the temporary directory
            shutil.rmtree(os.environ["TARGET_PATH"])

class SignalFit(MultiYearTask):
    
    def _requires_single(self):
        

        # Load the input configuration
        config = self.get_input_config()
        
        output_dir = self.get_output_dir()
            
        signal_input_path = glob.glob(config['inputFiles']['Trees2WS']+'/*')
        
        inOutSplittingFlag = config['trees2wsCfg']['doInOutSplitting']  or config['trees2wsCfg']['doDiffSplitting']
            
        tasks = []

        # Loop over a years era
        eras = allErasMap.get(f"{self.year}", [""])
        
        for currentEra in eras:
            
            era_suffix = "" if currentEra in ["", "None"] else currentEra
            
            if self.variable == "":
                input_path = os.path.join(output_dir, "Tree2WS", f"input_output_{self.year}{era_suffix}/ws_signal")
            else:
                input_path = os.path.join(output_dir, "Tree2WS", f"input_output_{self.variable}_{self.year}{era_suffix}/ws_signal")

            if currentEra not in ["", "None"]:
                currentConfig = config[f"signalScriptCfg_{self.year}_{era_suffix}"]
            else:
                currentConfig = config[f"signalScriptCfg_{self.year}"]

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # If proc/cat == auto. Extract processes and categories
            # Use allData.root from HiggsDNA to automatically determine categories
            data_input_path = config['inputFiles']['Trees2WSData']  
            if currentConfig['cats'] == "auto":
                currentConfig['cats'] = extractListOfCatsFromHiggsDNAAllData(data_input_path)
            currentConfig['nCats'] = len(currentConfig['cats'].split(","))

            if currentConfig['procs'] == "auto":
                currentConfig['procs'] = extractListOfProcsFromHiggsDNASignal(signal_input_path, self.variable, inOutSplittingFlag)
            currentConfig['nProcs'] = len(currentConfig['procs'].split(","))
            
            # Extract low and high MH values
            mps = []
            for mp in currentConfig['massPoints'].split(","): mps.append(int(mp))
            currentConfig['massLow'], currentConfig['massHigh'] = '%s'%min(mps), '%s'%max(mps)         
            
            tasks.append(SignalFitCategoryProcess.req(
                self,
                input_path=input_path, 
                output_dir=output_dir, 
                ext=currentConfig['ext'], 
                cats=currentConfig['cats'], 
                procs=currentConfig['procs'], 
                scales=currentConfig['scales'], 
                scalesCorr=currentConfig['scalesCorr'], 
                scalesGlobal=currentConfig['scalesGlobal'], 
                smears=currentConfig['smears'], 
                year=currentConfig['year'], 
                analysisXSBR=currentConfig['analysisXSBR'], 
                analysisRM=currentConfig['analysisRM'], 
                replacementThreshold=currentConfig['replacementThreshold'], 
                massPoints=currentConfig['massPoints'], 
                beamspotWidthData=currentConfig['beamspotWidthData'], 
                beamspotWidthMC=currentConfig['beamspotWidthMC'], 
                doPlots=currentConfig['doPlots'], 
                era=currentEra,
                workflow=currentConfig['execution'],
                slurm_partition=currentConfig['batchPartition'], 
                slurm_memory=currentConfig['batchMemory'], 
                slurm_max_runtime=currentConfig['batchMaxRuntime'], 
                htcondor_partition=currentConfig['batchPartition'], 
                htcondor_memory=currentConfig['batchMemory'], 
                htcondor_max_runtime=currentConfig['batchMaxRuntime']))
                
        return tasks
    
    def output(self):
        return self.input()

                
    def run(self):
        return True


class SignalTask(Task):
    ext = law.Parameter(default="packaged", description="Extension to be used for output folder naming")

    def get_ext(self):
        return f"{self.ext}_{self.year}"

class SignalMultiTask(MultiYearTask):
    ext = law.Parameter(default="packaged", description="Extension to be used for output folder naming")

    def get_ext(self):
        return f"{self.ext}_{self.year}"

class SignalPackagingCategory(SignalTask, HTCondorWorkflow, SlurmWorkflow, law.LocalWorkflow):#(Task): #(Task, HTCondorWorkflow, law.LocalWorkflow):
    exts = law.Parameter(default="earlyAnalysis", description="Extension to be used for output folder naming")
    cats = law.Parameter(description="List of categories separated with a comma.")
    era = law.Parameter(description="era")   
    massPoints = law.Parameter(description="Mass Points")
    mergeYears = law.Parameter(default=True, description="Flag if one should merge the years or eras.")
    requireSignalFit = law.Parameter(default=True, description="Require SignalFit before packaging.")
    
    
    htcondor_job_kwargs_submit = {"spool": True}
    
    # def requires(self):
    def workflow_requires(self):
        workflow_reqs = super().workflow_requires()

        tasks = {}

        if workflow_reqs:
            tasks.update(workflow_reqs)

        if not convert_boolean_string(self.requireSignalFit):
            return tasks

        for year in yearMap[self.year]:
            # Load the input configuration
            config = self.get_input_config(year=year)
            output_dir = self.get_output_dir()
                                    
            tasks[f"SignalFit_{year}"] = SignalFit.req(self, output_dir=output_dir, years=year)
                    
        return tasks
    
    
    def create_branch_map(self):
        # map branch indexes to ascii numbers from 97 to 122 ("a" to "z")
        nCats = len(self.cats.split(","))
        
        cat_list = [
            self.cats.split(",")[categoryIndex]
            for categoryIndex in range(nCats)
        ]
        
        branch_map = {i: cat for i, cat in enumerate(cat_list)}
        return branch_map

    def output(self):
        cat = self.branch_data
        
        signal_output = [os.path.join(self.output_dir, "Signal", f'outdir_{self.get_ext()}/CMS-HGG_sigfit_{self.ext}_{cat}.root')]
                
        outputFileTargets = []
            
        for _, current_output_path in enumerate(signal_output):
            outputFileTargets.append(law.LocalFileTarget(current_output_path))

        return outputFileTargets

    def run(self):
        cat = self.branch_data
        sys.path.append(os.path.dirname(os.path.abspath(__file__))+ "/tools")
        
        on_slurm_node = os.environ.get("SLURM_JOB_ID", False)

        if self.batch_flavor == "slurm/psi" and on_slurm_node:
            # Have to use /scratch/batch_username/ for slurm/psi
            if "/work" in self.output_dir:
                execute_command([f'mkdir -p {self.output_dir}/Signal/outdir_{self.get_ext()}/packageSignal'], shell=True)
            else:   
                execute_command([f'xrdfs root://t3dcachedb03.psi.ch:1094/ mkdir -p {self.output_dir}/outdir_{self.get_ext()}/packageSignal'], shell=True)

            os.environ["TARGET_PATH"] = f"/scratch/{os.environ['USER']}/{os.environ['SLURM_JOB_ID']}"
            execute_command([f'mkdir -p $TARGET_PATH/Signal/outdir_{self.get_ext()}/packageSignal'], shell=True)
            output_dir = os.environ["TARGET_PATH"]
            # os.chdir(os.path.join(os.environ["TARGET_PATH"], 'Combine', fitFolderName, 'impact'))
        else:
            execute_command([f'mkdir -p {self.output_dir}/Signal/outdir_{self.get_ext()}'], shell=True)
            output_dir = os.path.join(self.output_dir, "Signal")
            # os.chdir(os.path.join(self.output_dir, 'Combine', fitFolderName, 'impact'))
        
        if self.batch_flavor == "slurm/psi" and on_slurm_node:
            if self.variable == '':
                configYamlPath = os.path.join(os.environ["ANALYSIS_PATH"], f"config/{self.year}_inclusive.yml")
            else:
                configYamlPath = os.path.join(os.environ["ANALYSIS_PATH"], f"config/v{self.version}/{self.year}_{self.variable}.yml")
            
            #Load central config file
            with open(configYamlPath, 'r') as file:
                config = yaml.safe_load(file)
            eras = allErasMap.get(f"{self.year}", [""])
            
            for currentEra in eras:
                
                era_suffix = "" if currentEra in ["", "None"] else currentEra
        
                if currentEra not in ["", "None"]:
                    currentConfig = config[f"signalScriptCfg_{self.year}_{era_suffix}"]
                else:
                    currentConfig = config[f"signalScriptCfg_{self.year}"]
                # Have to copy over the input to the JOB directory
                # Don't forget to VOMS!
                if "/work" in self.output_dir:
                    slurm_copy_command = [
                        'cp', '-rf',
                        f"{self.output_dir}/outdir_{currentConfig['ext']}",
                        f"{os.environ['TARGET_PATH']}/"
                    ]
                else:
                    slurm_copy_command = [
                        'xrdcp', '-rf',
                        'root://t3dcachedb03.psi.ch:1094//'+f"{self.output_dir}/outdir_{currentConfig['ext']}",
                        f"{os.environ['TARGET_PATH']}/"
                    ]
                print(slurm_copy_command)
                execute_command(slurm_copy_command)

        script_path = os.path.join(os.environ["ANALYSIS_PATH"], "Signal/scripts/packageSignal.py")

        arguments = [
            "python3",
            script_path,
            "--cat", cat,
            "--outputExt", self.ext,
            "--exts", self.exts,
            "--outputDir", f"{output_dir}",
            "--year", f"{self.year}",
            "--massPoints", f"{self.massPoints}",
            "--mergeYears", f"{self.mergeYears}",
        ]
        command = arguments
        print(' '.join(command))
        try:
            result = subprocess.run(command, check=True, text=True, capture_output=True)
            print("Script output:", result.stdout)
            print("Script executed successfully.")
        except subprocess.CalledProcessError as e:
            print("Error executing script:", e.stderr)

        # Copy the files back to pnfs if we are on slurm/psi
        if self.batch_flavor == "slurm/psi" and on_slurm_node:
            # Have to copy over the output to the final directory
            # Don't forget to VOMS!
            if "/work" in self.output_dir:
                slurm_copy_command = [
                    'cp', '-rf',
                    f"{os.environ['TARGET_PATH']}/outdir_{self.get_ext()}/",
                    self.output_dir
                ]
            else:
                slurm_copy_command = [
                    'xrdcp', '-rf',
                    f"{os.environ['TARGET_PATH']}/outdir_{self.get_ext()}/",
                    'root://t3dcachedb03.psi.ch:1094//'+self.output_dir
                ]
            print(slurm_copy_command)
            execute_command(slurm_copy_command)
            # Clean up the temporary directory
            shutil.rmtree(os.environ["TARGET_PATH"])

class SignalPackaging(SignalMultiTask):

    def _requires_single(self):
        

        # Load the input configuration
        config = self.get_input_config()
        
        output_dir = self.get_output_dir()
            
        tasks = []
        
        packagedConfig = config[f"packaged_{self.year}"]
        
        mergeYears = packagedConfig['mergeYears']
        requireSignalFit = packagedConfig.get('requireSignalFit', True)
        signalFitDirs = packagedConfig.get('signalFitDirs', {})
        
        
        # Use allData.root from HiggsDNA to automatically determine categories
        data_input_path = config['inputFiles']['Trees2WSData']  
        if packagedConfig['cats'] == "auto":
            packagedConfig['cats'] = extractListOfCatsFromHiggsDNAAllData(data_input_path)
        packagedConfig['nCats'] = len(packagedConfig['cats'].split(","))

        # Extract low and high MH values
        mps = []
        for mp in packagedConfig['massPoints'].split(","): mps.append(int(mp))
        packagedConfig['massLow'], packagedConfig['massHigh'] = '%s'%min(mps), '%s'%max(mps)

        exts = []
            
        # Loop over a years era and extract the ext string in a list

        for yr in yearMap[self.year]:

            config = self.get_input_config(year=yr)

            eras = allErasMap.get(f"{yr}", [""])
            
            for currentEra in eras:
                
                era_suffix = "" if currentEra in ["", "None"] else currentEra

                if currentEra not in ["", "None"]:
                    currentConfig = config[f"signalScriptCfg_{yr}_{era_suffix}"]
                else:
                    currentConfig = config[f"signalScriptCfg_{yr}"]

                currentExt = currentConfig['ext']
                if currentExt in signalFitDirs:
                    exts.append(f"{currentExt}={signalFitDirs[currentExt]}")
                else:
                    exts.append(currentExt)
            
            exts_string = ''
            for i, currentExt in enumerate(exts):
                exts_string += currentExt
                if i < (len(exts) - 1):
                    exts_string += ','
                
        tasks.append(SignalPackagingCategory.req(
            self,
            output_dir=output_dir, 
            exts=exts_string, 
            ext=self.ext, 
            cats=packagedConfig['cats'], 
            era=currentEra,
            massPoints=packagedConfig['massPoints'], 
            mergeYears=mergeYears, 
            requireSignalFit=requireSignalFit, 
            workflow=packagedConfig['execution'], 
            slurm_partition=packagedConfig['batchPartition'], 
            slurm_memory=packagedConfig['batchMemory'], 
            slurm_max_runtime=packagedConfig['batchMaxRuntime'], 
            htcondor_partition=packagedConfig['batchPartition'], 
            htcondor_memory=packagedConfig['batchMemory'], 
            htcondor_max_runtime=packagedConfig['batchMaxRuntime']))
                
        return tasks

    
    def output(self):
        return self.input()
                
    def run(self):
        return True


class SignalPlotCategory(SignalTask, law.LocalWorkflow, HTCondorWorkflow, SlurmWorkflow):
    signal_dir = law.Parameter(default="", description="Signal directory containing outdir_<ext>")
    plot_input_dir = law.Parameter(default="", description="Input directory containing CMS-HGG_sigfit_<ext>_<cat>.root files")
    plot_output_dir = law.Parameter(default="", description="Output directory for RunPlotter plots")
    cats = law.Parameter(description="Comma separated list of categories")
    procs = law.Parameter(default="all", description="Processes passed to RunPlotter.py")
    years = law.Parameter(default="2022preEE", description="Years passed to RunPlotter.py")
    plot_years_separate = law.Parameter(default=True, description="Pass --plot-years-separate to RunPlotter.py")
    require_packaging = law.Parameter(default=True, description="Require SignalPackaging before plotting")


    htcondor_job_kwargs_submit = {"spool": True}

    def workflow_requires(self):
        workflow_reqs = super().workflow_requires()

        tasks = {}
        if workflow_reqs:
            tasks.update(workflow_reqs)

        if convert_boolean_string(self.require_packaging):
            tasks["SignalPackaging"] = SignalPackaging.req(self, years=self.year)

        return tasks

    def create_branch_map(self):
        cat_list = [cat.strip() for cat in self.cats.split(",") if cat.strip()]
        return {i: cat for i, cat in enumerate(cat_list)}

    def signal_path(self):
        if self.signal_dir != "":
            return self.signal_dir
        return os.path.join(os.environ["ANALYSIS_PATH"], "Signal")

    def input_path(self):
        if self.plot_input_dir != "":
            return self.plot_input_dir
        if self.output_dir != "":
            return os.path.join(self.output_dir, "Signal", f"outdir_{self.get_ext()}")
        return os.path.join(self.signal_path(), "Signal", f"outdir_{self.get_ext()}")

    def plot_path(self):
        if self.plot_output_dir != "":
            plot_dir = self.plot_output_dir
        elif self.output_dir != "":
            plot_dir = os.path.join(self.output_dir, "Signal", f"outdir_{self.get_ext()}", "Plots")
        else:
            plot_dir = os.path.join(self.signal_path(), "Signal", f"outdir_{self.get_ext()}", "Plots")

        if len(self.years.split(",")) > 1 and not convert_boolean_string(self.plot_years_separate):
            plot_dir = os.path.join(plot_dir, "noPlotYearsSeparate")

        return plot_dir

    def plot_suffix(self, cat):
        proc_ext = "" if self.procs == "all" else f"_{self.procs}"
        year_ext = "" if len(self.years.split(",")) > 1 else f"_{self.years}"
        return f"{cat}{proc_ext}{year_ext}"

    def output(self):
        cat = self.branch_data
        plot_dir = self.plot_path()
        suffix = self.plot_suffix(cat)
        return [
            law.LocalFileTarget(os.path.join(plot_dir, f"smodel_{suffix}.pdf")),
            law.LocalFileTarget(os.path.join(plot_dir, f"smodel_{suffix}.png")),
        ]

    def run(self):
        cat = self.branch_data
        signal_dir = self.signal_path()
        safe_mkdir(self.plot_path())

        env = os.environ.copy()
        env["ANALYSIS_PATH"] = os.path.dirname(signal_dir)

        script_path = os.path.join(signal_dir, "RunPlotter.py")
        command = [
            "python3",
            script_path,
            "--procs", self.procs,
            "--years", self.years,
            "--cats", cat,
            "--ext", self.ext,
            "--inputDir", self.input_path(),
            "--outputDir", self.plot_path(),
        ]

        if convert_boolean_string(self.plot_years_separate):
            command.append("--plot-years-separate")

        print(" ".join(command))
        result = subprocess.run(
            command,
            check=True,
            text=True,
            capture_output=True,
            cwd=signal_dir,
            env=env,
        )
        print("Script output:", result.stdout)


class SignalPlot(SignalTask):
    signal_dir = law.Parameter(default="", description="Signal directory containing outdir_<ext>")
    plot_input_dir = law.Parameter(default="", description="Input directory containing CMS-HGG_sigfit_<ext>_<cat>.root files")
    plot_output_dir = law.Parameter(default="", description="Output directory for RunPlotter plots")
    cats = law.Parameter(default="", description="Comma separated list of categories. If empty, uses packaged_<year>.cats from config.")
    procs = law.Parameter(default="all", description="Processes passed to RunPlotter.py")
    years = law.Parameter(default="2022preEE", description="Years passed to RunPlotter.py")
    plot_years_separate = law.Parameter(default=True, description="Pass --plot-years-separate to RunPlotter.py")
    require_packaging = law.Parameter(default=True, description="Require SignalPackaging before plotting")



    def packaged_config(self):
        config = self.get_input_config()

        packagedConfig = config[f"packaged_{self.year}"]

        if self.cats == "":
            cats = packagedConfig["cats"]
            if cats == "auto":
                data_input_path = config['inputFiles']['Trees2WSData']
                cats = extractListOfCatsFromHiggsDNAAllData(data_input_path)
        else:
            cats = self.cats

        ext = self.ext if self.ext != "" else f"packaged{packagedConfig['ext']}"
        output_dir = self.output_dir if self.output_dir != "" else os.path.join(config['outputFolder'], f'v{self.version}')

        return cats, ext, output_dir

    def requires(self):
        cats, ext, output_dir = self.packaged_config()

        return SignalPlotCategory.req(
            self,
            ext=ext,
            cats=cats,
            output_dir=output_dir,
        )

    def output(self):
        return self.input()

    def run(self):
        return True


class SignalPlotEverything(SignalMultiTask):
    signal_dir = law.Parameter(default="", description="Signal directory containing outdir_<ext>")
    plot_input_dir = law.Parameter(default="", description="Input directory containing CMS-HGG_sigfit_<ext>_<cat>.root files")
    plot_output_dir = law.Parameter(default="", description="Output directory for RunPlotter plots")
    ext = law.Parameter(default="", description="Extension used by RunPlotter.py. If empty, uses packaged_<year>.ext from config.")
    procs = law.Parameter(default="all", description="Processes passed to RunPlotter.py")
    require_packaging = law.Parameter(default=True, description="Require SignalPackaging before plotting")



    def plot_years(self):
        config = self.get_input_config()
        
        years = []

        for year in yearMap[self.year]:
            eras = allErasMap.get(f"{year}", [""])
            config = self.get_input_config(year=year)
            for currentEra in eras:
                era_suffix = "" if currentEra in ["", "None"] else currentEra
                if currentEra not in ["", "None"]:
                    config_key = f"signalScriptCfg_{year}_{era_suffix}"
                else:
                    config_key = f"signalScriptCfg_{year}"
                if config_key in config and "year" in config[config_key]:
                    years.append(config[config_key]["year"])

        if not years:
            for config_key in sorted(config):
                if config_key.startswith(f"signalScriptCfg_{self.year}") and "year" in config[config_key]:
                    years.append(config[config_key]["year"])

        years = list(dict.fromkeys(years))

        if len(years) > 1:
            return years + [",".join(years)]
        return years

    def _requires_single(self):
        tasks = []
        cat_modes = ["", "all", "wall"]
        print(f"SignalPlotEverything: Plotting for years: {self.plot_years()} and categories: {cat_modes}")
        for years in self.plot_years():
            for cats in cat_modes:
                plot_years_separate_options = [True]
                if len(years.split(",")) > 1:
                    plot_years_separate_options.append(False)

                for plot_years_separate in plot_years_separate_options:
                    tasks.append(
                        SignalPlot.req(
                            self,
                            cats=cats,
                            years=years,
                            plot_years_separate=plot_years_separate,
                        )
                    )

        return tasks

    def output(self):
        return self.input()

    def run(self):
        return True
