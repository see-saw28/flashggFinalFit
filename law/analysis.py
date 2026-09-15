import law
import luigi
import os
import yaml
import errno
import shutil
import subprocess

from commonTools import *
from commonObjects import *

from Datacard.law_datacard import *
from Background.law_background import *
from Trees2WS.law_trees2ws import *
from Signal.law_signal import *
from Combine.law_combine import *
from Plots.Spectra.law_spectra import *
from pathlib import Path

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

def is_signal_only_datacard(config):
    return convert_boolean_string(config.get("datacard_yields", {}).get("skipBkg", False))

HIGGS_MASS = "125.07"

class FinalFits(Task):

    combined = luigi.BoolParameter(default=False)
    yearly = luigi.BoolParameter(default=False)
    
    unblinded_fits = luigi.BoolParameter(default=False)
    unblinded_stage_one = luigi.BoolParameter(default=False)
    unblinded_stage_two = luigi.BoolParameter(default=False)
    unblinded_stage_three = luigi.BoolParameter(default=False)
    unblinded_covcorr = luigi.BoolParameter(default=False)
    pvalue = luigi.BoolParameter(default=False)
    asimov_fits = luigi.BoolParameter(default=False)
    asimov_impacts = luigi.BoolParameter(default=False)
    asimov_covcorr = luigi.BoolParameter(default=False)
    asimov_mgg = luigi.BoolParameter(default=False)
    unblinded_diff_spectra = luigi.BoolParameter(default=False)
    asimov_diff_spectra = luigi.BoolParameter(default=False)
    batch_system = law.Parameter(default="local")
    batch_flavor = law.Parameter(default="local")

    def requires(self):
        years = yearMap[self.year]
        multi_year = len(years) > 1  # when combining years, only force per-year datacards

        if not self.combined and not self.yearly and multi_year:
            print("Please set either combined or yearly to True.")
            exit(1)

        return {y:
            FinalFitsYear.req(
                self,
                year=y,
                datacard_only=multi_year and not(self.unblinded_fits or self.unblinded_stage_one or self.unblinded_stage_two or self.unblinded_stage_three or self.unblinded_covcorr or self.pvalue or self.asimov_fits or self.asimov_impacts or self.asimov_covcorr or self.unblinded_diff_spectra or self.asimov_diff_spectra or self.asimov_mgg),
            )
            for y in (years if self.yearly else []) + ([self.year] if (self.combined or not multi_year) else [])
        }
        

    def run(self):
        return True


    
    def output(self):
        return self.input()



class FinalFitsYear(Task):

    datacard_only = luigi.BoolParameter(default=False, description="If True, stop after building datacards/text2workspace")
    
    # Unblinded fits and impacts
    unblinded_fits = luigi.BoolParameter(default=False, description="Produce unblinded fits")
    unblinded_stage_one = luigi.BoolParameter(default=False, description="Produce Stage 1 unblinded results")
    unblinded_stage_two = luigi.BoolParameter(default=False, description="Produce Stage 2 unblinded results")
    unblinded_stage_three = luigi.BoolParameter(default=False, description="Produce Stage 3 unblinded results")
    unblinded_covcorr = luigi.BoolParameter(default=False, description="Produce the Unblinded covariance and correlation matrices")
    pvalue = luigi.BoolParameter(default=False, description="Produce p-value for the unblinded fit")
    
    # Asimov fits and impacts
    asimov_fits = luigi.BoolParameter(default=False, description="Produce the Asimov fits")
    asimov_impacts = luigi.BoolParameter(default=False, description="Produce the Asimov impacts")
    asimov_covcorr = luigi.BoolParameter(default=False, description="Produce the Asimov covariance and correlation matrices")
    asimov_mgg = luigi.BoolParameter(default=False, description="Produce the Asimov mgg distribution for the given variable")

    # Differentials
    unblinded_diff_spectra = luigi.BoolParameter(default=False, description="Produce unblinded differential spectra for the given variable")
    asimov_diff_spectra = luigi.BoolParameter(default=False, description="Produce Asimov differential spectra for the given variable")
    
    batch_system = law.Parameter(default="local", description="Batch system to use")
    
    def requires(self):
        # req() is defined on all tasks and handles the passing of all parameter values that are
        # common between the required task and the instance (self)
        
        tasks = {}
        
        output_dir = self.get_output_dir()
        config = self.get_input_config()

        if self.datacard_only:
            if is_signal_only_datacard(config):
                yieldsConfig = config["datacard_yields"]
                tasks["MakeDatacard"] = MakeDatacard.req(
                    self,
                    output_dir=output_dir,
                    workflow=yieldsConfig["execution"],
                    slurm_partition=yieldsConfig['batchPartition'],
                    slurm_memory=yieldsConfig['batchMemory'],
                    slurm_max_runtime=yieldsConfig['batchMaxRuntime'],
                    htcondor_partition=yieldsConfig['batchPartition'],
                    htcondor_memory=yieldsConfig['batchMemory'],
                    htcondor_max_runtime=yieldsConfig['batchMaxRuntime'],
                )
            else:
                tasks["RunT2WS"] = RunText2Workspace.req(
                    self,
                    output_dir=output_dir,
                    years=self.year,
                )
            return tasks

        if self.unblinded_fits:
            tasks["CreateLikelihoodFitObserved"] = CreateLikelihoodFitWrapper.req(
                self, output_dir=output_dir, years=self.year, is_postfit=True
            )
        if self.unblinded_stage_one:
            impactConfig = config["combine_impacts"]
            tasks["GoodnessOfFit"] = UnblindedGoodnessOfFit.req(self, output_dir=output_dir)
            tasks["UnblindedImpactThirdStep"] = UnblindedImpactThirdStep.req(self, output_dir=output_dir)#, workflow=impactConfig["execution"], slurm_partition=impactConfig['batchPartition'], slurm_memory=impactConfig['batchMemory'], slurm_max_runtime=impactConfig['batchMaxRuntime'], htcondor_partition=impactConfig['batchPartition'], htcondor_memory=impactConfig['batchMemory'], htcondor_max_runtime=impactConfig['batchMaxRuntime'])
        if self.unblinded_stage_two:
            impactConfig = config["combine_impacts"]
            tasks["GoodnessOfFit"] = UnblindedGoodnessOfFit.req(self, output_dir=output_dir)
            tasks["UnblindedImpactThirdStep"] = UnblindedImpactThirdStep.req(self, output_dir=output_dir)#, workflow=impactConfig["execution"], slurm_partition=impactConfig['batchPartition'], slurm_memory=impactConfig['batchMemory'], slurm_max_runtime=impactConfig['batchMaxRuntime'], htcondor_partition=impactConfig['batchPartition'], htcondor_memory=impactConfig['batchMemory'], htcondor_max_runtime=impactConfig['batchMaxRuntime'])
            tasks["MggDistribution"] = MggDistribution.req(self, output_dir=output_dir, years=self.year, is_postfit=True)
        if self.unblinded_stage_three:
            impactConfig = config["combine_impacts"]
            tasks["GoodnessOfFit"] = UnblindedGoodnessOfFit.req(self, output_dir=output_dir)
            tasks["UnblindedImpactThirdStep"] = UnblindedImpactThirdStep.req(self, output_dir=output_dir)#, workflow=impactConfig["execution"], slurm_partition=impactConfig['batchPartition'], slurm_memory=impactConfig['batchMemory'], slurm_max_runtime=impactConfig['batchMaxRuntime'], htcondor_partition=impactConfig['batchPartition'], htcondor_memory=impactConfig['batchMemory'], htcondor_max_runtime=impactConfig['batchMaxRuntime'])
            tasks["MggDistribution"] = MggDistribution.req(self, output_dir=output_dir, years=self.year, is_postfit=True)
            tasks["CreateLikelihoodFitObserved"] = CreateLikelihoodFitWrapper.req(
                self, output_dir=output_dir, years=self.year, is_postfit=True
            )
        if self.unblinded_covcorr:
            hesseConfig = config["combine_hesse"]
            tasks["UnblindedCovCorr"] = UnblindedCovCorr.req(self, output_dir=output_dir, workflow=hesseConfig["execution"], slurm_partition=hesseConfig['batchPartition'], slurm_memory=hesseConfig['batchMemory'], slurm_max_runtime=hesseConfig['batchMaxRuntime'], htcondor_partition=hesseConfig['batchPartition'], htcondor_memory=hesseConfig['batchMemory'], htcondor_max_runtime=hesseConfig['batchMaxRuntime'])
        if self.pvalue:
            hesseConfig = config["combine_hesse"]
            tasks["PValueCalculation"] = PValueCalculation.req(self, output_dir=output_dir, workflow=self.batch_system, slurm_partition=hesseConfig['batchPartition'], slurm_memory=hesseConfig['batchMemory'], slurm_max_runtime=hesseConfig['batchMaxRuntime'], htcondor_partition=hesseConfig['batchPartition'], htcondor_memory=hesseConfig['batchMemory'], htcondor_max_runtime=hesseConfig['batchMaxRuntime'])
        if self.asimov_fits:
            tasks["CreateLikelihoodFitAsimov"] = CreateLikelihoodFitWrapper.req(self, years=self.year, output_dir=output_dir, is_postfit=False)
            tasks["CreateFitPerCat"] = CreateFitPerCat.req(self, years=self.year, output_dir=output_dir, is_postfit=False)
        if self.asimov_impacts:
            tasks["AsimovImpactThirdStep"] = AsimovImpactThirdStep.req(self, output_dir=output_dir, workflow=self.batch_system)
        if self.asimov_covcorr:
            hesseConfig = config["combine_hesse"]
            tasks["AsimovCovCorr"] = AsimovCovCorr.req(self, output_dir=output_dir, workflow=hesseConfig["execution"], slurm_partition=hesseConfig['batchPartition'], slurm_memory=hesseConfig['batchMemory'], slurm_max_runtime=hesseConfig['batchMaxRuntime'], htcondor_partition=hesseConfig['batchPartition'], htcondor_memory=hesseConfig['batchMemory'], htcondor_max_runtime=hesseConfig['batchMaxRuntime'])
        if self.asimov_mgg:
            tasks["MggDistribution"] = MggDistribution.req(self, output_dir=output_dir,years=self.year)#workflow=mggConfig["execution"], slurm_partition=mggConfig['batchPartition'], slurm_memory=mggConfig['batchMemory'], slurm_max_runtime=mggConfig['batchMaxRuntime'], htcondor_partition=mggConfig['batchPartition'], htcondor_memory=mggConfig['batchMemory'], htcondor_max_runtime=mggConfig['batchMaxRuntime'])

        if self.asimov_diff_spectra and (self.variable != ''):
            tasks["CreateAsimovDiffSpectra"] = CreateDiffSpectra.req(self, output_dir=output_dir, batch_system=self.batch_system, is_unblinded=False)
        if self.unblinded_diff_spectra and (self.variable != ''):
            tasks["CreateUnblindedDiffSpectra"] = CreateDiffSpectra.req(self, output_dir=output_dir, batch_system=self.batch_system, is_unblinded=True)
        if (self.asimov_diff_spectra or self.unblinded_diff_spectra) and (self.variable == ''):
            print("Differential spectra can only be created for a specific variable. Please set the variable parameter to a valid value.")
            exit(1)
        if tasks == {}:
            print("No final fit tasks selected. Please set the appropriate parameters to True.")
            exit(1)

        return tasks

    def output(self):
        # returns output folder

        return self.input()
    
    def run(self):
        
        return True
