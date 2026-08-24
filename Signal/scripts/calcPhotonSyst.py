# Script to calculate photon systematics
# * Run script once per category, loops over signal processes
# * Output is pandas dataframe 

print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ HGG PHOTON SYST CALCULATOR ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ")
import ROOT
import pandas as pd
import pickle
import os, sys
from optparse import OptionParser
import glob
import re

# From tools
from plottingTools import * #getEffSigma function
from commonTools import *
from commonObjects import *

def leave():
  print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ HGG PHOTON SYST CALCULATOR (END) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ")
  exit(0)

def get_options():
  parser = OptionParser()
  parser.add_option("--xvar", dest='xvar', default='CMS_hgg_mass', help="Observable")
  parser.add_option("--cat", dest='cat', default='', help="RECO category")
  parser.add_option("--procs", dest='procs', default='', help="Signal processes")
  parser.add_option("--ext", dest='ext', default='', help="Extension")
  parser.add_option("--inputWSDir", dest='inputWSDir', default='', help="Input flashgg WS directory")
  parser.add_option("--outputDir", dest='outputDir', default=swd__, help="Output directory")
  parser.add_option("--scales", dest='scales', default='', help="Photon shape systematics: scales")
  parser.add_option("--scalesCorr", dest='scalesCorr', default='', help='Photon shape systematics: scalesCorr')
  parser.add_option("--scalesGlobal", dest='scalesGlobal', default='', help='Photon shape systematics: scalesGlobal')
  parser.add_option("--smears", dest='smears', default='', help='Photon shape systematics: smears')
  parser.add_option("--nBins", dest='nBins', default=80, type='int', help='Number of bins in histograms')
  parser.add_option("--thresholdMean", dest='thresholdMean', default=0.05, type='float', help='Reject mean variations if larger than thresholdMean')
  parser.add_option("--thresholdSigma", dest='thresholdSigma', default=0.5, type='float', help='Reject mean variations if larger than thresholdSigma')
  parser.add_option("--thresholdRate", dest='thresholdRate', default=0.05, type='float', help='Reject mean variations if larger than thresholdRate')
  parser.add_option("--doPlots", dest='doPlots', default=False, action='store_true', help='Save nominal/up/down histogram plots for each photon systematic')
  parser.add_option("--plotFormat", dest='plotFormat', default='png,pdf', help='Comma-separated output formats for --doPlots')
  return parser.parse_args()
(opt,args) = get_options()

if opt.doPlots:
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt

# RooRealVar to fill histograms
mgg = ROOT.RooRealVar(opt.xvar,opt.xvar,125)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Function to extact histograms from WS
def getHistograms( _ws, _nominalDataName, _sname ):
  _hists = {}
  # Define histograms
  for htype in ['nominal','up','down']:
    if htype == 'nominal': _hists[htype] = ROOT.TH1F(htype,htype,opt.nBins,100,180)
    else: _hists[htype] = ROOT.TH1F("%s_%s"%(_sname,htype),"%s_%s"%(_sname,htype),opt.nBins,100,180)
  # Extract nominal RooDataSet and syst RooDataHists
  rds_nominal = _ws.data(_nominalDataName)
  rdh_up = _ws.data("%s_%sUp01sigma"%(_nominalDataName,_sname))
  rdh_down = _ws.data("%s_%sDown01sigma"%(_nominalDataName,_sname))
  # Check if not NONE type and fill histograms
  if rds_nominal: rds_nominal.fillHistogram(_hists['nominal'],ROOT.RooArgList(mgg))
  else:
    print(" --> [ERROR] Could not extract nominal RooDataSet: %s. Leaving"%_nominalDataName)
    sys.exit(1)
  if rdh_up: rdh_up.fillHistogram(_hists['up'],ROOT.RooArgList(mgg))
  else:
    print(" --> [ERROR] Could not extract RooDataHist (%s,up) for %s. Leaving"%(_sname,_nominalDataName))
    sys.exit(1)
  if rdh_down: rdh_down.fillHistogram(_hists['down'],ROOT.RooArgList(mgg))
  else:
    print(" --> [ERROR] Could not extract RooDataHist (%s,down) for %s. Leaving"%(_sname,_nominalDataName))
    sys.exit(1) 
  return _hists

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Functions to extract mean, sigma and rate variations
def getMeanVar(_hists):
  mu, muVar = {}, {}
  for htype,h in _hists.items(): mu[htype] = h.GetMean()
  if mu['nominal']==0: return 0
  for htype in ['up','down']: muVar[htype] = (mu[htype]-mu['nominal'])/mu['nominal']
  x = (abs(muVar['up'])+abs(muVar['down']))/2
  # Check for NaN
  if x!=x: return 0
  else: return min(x,opt.thresholdMean)

def getSigmaVar(_hists):
  sigma, sigmaVar = {}, {}
  for htype,h in _hists.items(): sigma[htype] = getEffSigma(h)
  if sigma['nominal']==0: return 0
  for htype in ['up','down']: sigmaVar[htype] = (sigma[htype]-sigma['nominal'])/sigma['nominal']
  x = (abs(sigmaVar['up'])+abs(sigmaVar['down']))/2
  if x!=x: return 0
  else: return min(x,opt.thresholdSigma)

def getRateVar(_hists):
  rate, rateVar = {}, {}
  for htype,h in _hists.items(): rate[htype] = h.Integral()
  # Shape variations can both be one sided therefore use midpoint as nominal
  rate['midpoint'] = 0.5*(rate['up']+rate['down'])
  if rate['midpoint']==0: return 0
  for htype in ['up','down']: rateVar[htype] = (rate[htype]-rate['midpoint'])/rate['midpoint']
  x = (abs(rateVar['up'])+abs(rateVar['down']))/2
  if x!=x: return 0
  else: return min(x,opt.thresholdRate)

def getFinalSystName(_s, _stype):
  outputNuisanceExt = "_%s"%outputNuisanceExtMap[_stype] if outputNuisanceExtMap[_stype] != "" else ""
  return "%s%s"%(_s,outputNuisanceExt)

def sanitizeName(_name):
  return re.sub(r'[^A-Za-z0-9_.-]+','_',_name)

def histToArrays(_h):
  edges = [_h.GetBinLowEdge(1)]
  values = []
  for ibin in range(1,_h.GetNbinsX()+1):
    values.append(_h.GetBinContent(ibin))
    edges.append(_h.GetBinLowEdge(ibin)+_h.GetBinWidth(ibin))
  return edges, values

def saveSystPlot(_hists, _proc, _cat, _stype, _sname, _finalName, _plotDir, _savedInfo):
  colors = {'down':'#1f77b4','nominal':'#222222','up':'#d62728'}
  linestyles = {'down':'--','nominal':'-','up':':'}
  labels = {'down':'Down','nominal':'Nominal','up':'Up'}
  fig, ax = plt.subplots(figsize=(8,6))
  for htype in ['down','nominal','up']:
    edges, values = histToArrays(_hists[htype])
    label = "%s: μ=%.3f, σ=%.3f, rate=%.4g"%(labels[htype],_hists[htype].GetMean(),getEffSigma(_hists[htype]),_hists[htype].Integral())
    ax.stairs(values, edges, label=label, color=colors[htype], linestyle=linestyles[htype], linewidth=1.8)
  ax.set_xlabel(opt.xvar)
  ax.set_ylabel("Events")
  ax.set_xlim(115, 140)
  ax.text(0.08,0.96,f"{_proc}, {_cat},\n{_stype}, {_sname}", transform=ax.transAxes, ha='left', va='top')
  ax.legend(loc='best', frameon=False, fontsize=9, 
            title=f"{_finalName}:\n Δμ={_savedInfo['mean']*100:.3f} %, Δσ={_savedInfo['sigma']*100:.3f} %, Δrate={_savedInfo['rate']*100:.3f} %")
  fig.tight_layout()
  for fmt in opt.plotFormat.split(","):
    if fmt == '': continue
    fig.savefig("%s/%s_%s_%s.%s"%(_plotDir,sanitizeName(_cat),sanitizeName(_proc),sanitizeName(_sname),fmt))
  plt.close(fig)

def saveSystPlotsFromData(_data, _plotDir):
  if not os.path.isdir(_plotDir): os.makedirs(_plotDir)
  print(" --> Saving photon systematic plots in: %s"%_plotDir)
  for ir,r in _data.iterrows():
    f = ROOT.TFile(r['inputWSFile'])
    inputWS = f.Get(inputWSName__)
    for stype in ['scales','scalesCorr','smears']:
      for s in getattr(opt,stype).split(","):
        if s == '': continue
        finalName = getFinalSystName(s,stype)
        if "%s_mean"%finalName not in _data.columns: continue
        sname = "%s%s"%(inputNuisanceExtMap[stype],s)
        hists = getHistograms(inputWS,r['nominalDataName'],sname)
        savedInfo = {
          'mean': r["%s_mean"%finalName],
          'sigma': r["%s_sigma"%finalName],
          'rate': r["%s_rate"%finalName]
        }
        saveSystPlot(hists,r['proc'],opt.cat,stype,sname,finalName,_plotDir,savedInfo)
        for h in hists.values(): h.Delete()
    inputWS.Delete()
    f.Close()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define dataFrame
columns_data = ['proc','cat','inputWSFile','nominalDataName']
for stype in ['scales','scalesCorr','smears']:
  systs = getattr( opt, stype )
  for s in systs.split(","):
    if s == '': continue
    for x in ['mean','sigma','rate']: 
      outputNuisanceExt = "_%s"%outputNuisanceExtMap[stype] if outputNuisanceExtMap[stype] != "" else ""
      columns_data.append("%s%s_%s"%(s,outputNuisanceExt,x))
data = pd.DataFrame( columns=columns_data ) 

# Loop over processes and add row to dataframe
for _proc in opt.procs.split(","):
  # Glob M125 filename
  _WSFileName = glob.glob(f"{opt.inputWSDir}/output*M125*{_proc}.root")[0]
  # Hard-coded for fiducial inclusve with in/out, should maybe be adjusted
  if (len(_proc.split("_")) <= 2) and (_proc.split("_")[-1] in ["in", "out"]):
    _nominalDataName = "%s_%s_125_%s_%s"%(procToData(_proc.split("_")[0]),procToData(_proc.split("_")[-1]),sqrts__,opt.cat)
  else:
    _nominalDataName = "%s_125_%s_%s"%(procToData(_proc.split("_")[0]),sqrts__,opt.cat)
  #data = data.append({'proc':_proc,'cat':opt.cat,'inputWSFile':_WSFileName,'nominalDataName':_nominalDataName}, ignore_index=True, sort=False)
  data = pd.concat([data,pd.DataFrame([{'proc':_proc,'cat':opt.cat,'inputWSFile':_WSFileName,'nominalDataName':_nominalDataName}])], ignore_index=True, sort=False)

# Loop over rows in dataFrame and open ws
for ir,r in data.iterrows():

  print(" --> Processing (%s,%s)"%(r['proc'],opt.cat))

  # Open ROOT file and extract workspace
  f = ROOT.TFile(r['inputWSFile'])
  inputWS = f.Get(inputWSName__)
 
  # Loop over scale and smear systematics
  for stype in ['scales','scalesCorr','smears']:
    for s in getattr(opt,stype).split(","):
      if s == '': continue
      # Note: Here was an else statement from JLS, include back in if the code does not run
      sname = "%s%s"%(inputNuisanceExtMap[stype],s)
      #print("    * Systematic = %s (%s)"%(sname,stype))
      hists = getHistograms(inputWS,r['nominalDataName'],sname)
      # If nominal yield = 0:
      if hists['nominal'].Integral() == 0: _meanVar, _sigmaVar, _rateVar = 0, 0, 0
      else:
        _meanVar = getMeanVar(hists)
        _sigmaVar = getSigmaVar(hists)
        _rateVar = getRateVar(hists)
      # Add values to dataFrame
      outputNuisanceExt = "_%s"%outputNuisanceExtMap[stype] if outputNuisanceExtMap[stype] != "" else ""
      data.at[ir,'%s%s_mean'%(s,outputNuisanceExt)] = _meanVar
      data.at[ir,'%s%s_sigma'%(s,outputNuisanceExt)] = _sigmaVar
      data.at[ir,'%s%s_rate'%(s,outputNuisanceExt)] = _rateVar

      # Delete histograms
      for h in hists.values(): h.Delete()

  # Delete ws and close file
  inputWS.Delete()
  f.Close()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Output dataFrame as pickle file to be read in by signalFit.py
if not os.path.isdir("%s/outdir_%s"%(opt.outputDir,opt.ext)): os.system("mkdir %s/outdir_%s"%(opt.outputDir,opt.ext))
if not os.path.isdir("%s/outdir_%s/calcPhotonSyst"%(opt.outputDir,opt.ext)): os.system("mkdir %s/outdir_%s/calcPhotonSyst"%(opt.outputDir,opt.ext))
if not os.path.isdir("%s/outdir_%s/calcPhotonSyst/pkl"%(opt.outputDir,opt.ext)): os.system("mkdir %s/outdir_%s/calcPhotonSyst/pkl"%(opt.outputDir,opt.ext))
with open("%s/outdir_%s/calcPhotonSyst/pkl/%s.pkl"%(opt.outputDir,opt.ext,opt.cat),"wb") as f: pickle.dump(data,f) 
print(" --> Successfully saved photon systematics as pkl file: %s/outdir_%s/calcPhotonSyst/pkl/%s.pkl"%(opt.outputDir,opt.ext,opt.cat))

if opt.doPlots:
  plotDir = "%s/outdir_%s/calcPhotonSyst/plots"%(opt.outputDir,opt.ext)
  saveSystPlotsFromData(data,plotDir)
