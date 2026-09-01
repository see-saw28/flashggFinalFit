#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import print_function

import argparse
import json
import math
import os
from functools import partial

import ROOT
import CombineHarvester.CombineTools.plotting as plot

ROOT.PyConfig.IgnoreCommandLineOptions = True
ROOT.gROOT.SetBatch(ROOT.kTRUE)

plot.ModTDRStyle(width=1100, height=650, l=0.26, r=0.05, t=0.08, b=0.12)
ROOT.gStyle.SetNdivisions(510, "XYZ")
ROOT.gStyle.SetMarkerSize(1.2)

NAMECOUNTER = 0


def read_scan(scan, param, files, ycut):
    goodfiles = [f for f in files if plot.TFileIsGood(f)]
    if not goodfiles:
        raise RuntimeError("No good ROOT files for scan %s: %s" % (scan, files))
    limit = plot.MakeTChain(goodfiles, "limit")
    graph = plot.TGraphFromTree(limit, param, "2*deltaNLL", "quantileExpected > -1.5")
    graph.SetName(scan)
    graph.Sort()
    plot.RemoveGraphXDuplicates(graph)
    plot.RemoveGraphYAbove(graph, ycut)
    return graph


def eval_spline(obj, x, params):
    return obj.Eval(x[0])


def build_scan(name, param, files, yvals, ycut):
    graph = read_scan(name, param, files, ycut)
    if graph.GetN() <= 1:
        graph.Print()
        raise RuntimeError("Attempting to build %s from an empty or single-point scan" % name)

    bestfit = None
    min_y = None
    for i in range(graph.GetN()):
        y = graph.GetY()[i]
        if y == 0.:
            bestfit = graph.GetX()[i]
            break
        if min_y is None or y < min_y[1]:
            min_y = (graph.GetX()[i], y)
    if bestfit is None:
        bestfit = min_y[0]
        print("Warning: scan %s has no exact y=0 point, using minimum sampled point %.5g" % (name, bestfit))

    spline = ROOT.TSpline3("spline3_%s" % name, graph)
    global NAMECOUNTER
    func_method = partial(eval_spline, spline)
    func = ROOT.TF1("splinefn%d" % NAMECOUNTER, func_method, graph.GetX()[0], graph.GetX()[graph.GetN() - 1], 1)
    func._method = func_method
    NAMECOUNTER += 1

    crossings = {}
    val = None
    for yval in yvals:
        crossings[yval] = plot.FindCrossingsWithSpline(graph, func, yval)
        for cr in crossings[yval]:
            cr["contains_bf"] = cr["lo"] <= bestfit and cr["hi"] >= bestfit
    for cr in crossings[yvals[0]]:
        if cr["contains_bf"]:
            val = (bestfit, cr["hi"] - bestfit, cr["lo"] - bestfit)
            break
    if val is None:
        raise RuntimeError("Could not find a 1 sigma crossing containing the best fit for %s" % name)

    return {
        "graph": graph,
        "spline": spline,
        "func": func,
        "crossings": crossings,
        "val": val,
    }


def quad_subtract(total, stat, label):
    out = []
    for t, s, side in zip(total, stat, ["hi", "lo"]):
        if abs(s) > abs(t):
            print("Warning: stat uncertainty larger than total for %s %s; syst set to 0" % (label, side))
            out.append(0.0)
        else:
            out.append(math.sqrt(t * t - s * s))
    return tuple(out)


def parse_scan_arg(scan_arg):
    parts = scan_arg.split(":")
    if len(parts) not in (2, 3):
        raise RuntimeError("--scan must be LABEL:FULL.root or LABEL:FULL.root:STAT.root, got %s" % scan_arg)
    return {
        "label": parts[0],
        "full": parts[1],
        "stat": parts[2] if len(parts) == 3 and parts[2] else None,
    }


def load_scans(args):
    scans = [parse_scan_arg(x) for x in args.scan]
    if args.input_json:
        with open(args.input_json) as handle:
            payload = json.load(handle)
        for item in payload:
            scans.append({
                "label": item["label"],
                "full": item["full"],
                "stat": item.get("stat"),
            })
    if not scans:
        raise RuntimeError("Provide at least one --scan or --input-json entry")
    return scans


def format_unc(val, precision):
    return ("%.*f^{#plus %.*f}_{#minus %.*f}" %
            (precision, val[0], precision, abs(val[1]), precision, abs(val[2])))


def main():
    parser = argparse.ArgumentParser(
        description="Make a horizontal summary plot from one or more 1D Asimov scans."
    )
    parser.add_argument("--scan", action="append", default=[],
                        help="Line to draw: LABEL:FULL.root or LABEL:FULL.root:STAT.root. Repeat for more lines.")
    parser.add_argument("--input-json", default=None,
                        help="Optional JSON list with entries containing label, full, and optional stat.")
    parser.add_argument("--POI", default="MH", help="Parameter of interest stored in the limit tree")
    parser.add_argument("--output", "-o", default="asimov_scan_summary", help="Output name without extension")
    parser.add_argument("--translate", default=None, help="JSON file translating the POI name")
    parser.add_argument("--x-title", default=None, help="Override x-axis title")
    parser.add_argument("--x-min", type=float, default=None)
    parser.add_argument("--x-max", type=float, default=None)
    parser.add_argument("--y-cut", type=float, default=7.0, help="Drop scan points above this 2*deltaNLL value")
    parser.add_argument("--show-values", action="store_true", help="Draw central value and uncertainties on each row")
    parser.add_argument("--show-breakdown", action="store_true",
                        help="When stat scans are provided, draw stat bars and print stat/syst components")
    parser.add_argument("--band", default=None,
                        help="Draw a vertical band as CENTER:LO:HI, where LO/HI are positive uncertainties")
    parser.add_argument("--band-from", type=int, default=None,
                        help="Draw the vertical band from the 1-based scan index")
    parser.add_argument("--split-after", type=int, default=None,
                        help="Draw a horizontal separator after this many rows")
    parser.add_argument("--cms-label", default="Internal")
    parser.add_argument("--lumi-label", default="13 TeV")
    parser.add_argument("--precision", type=int, default=3)
    parser.add_argument("--marker-color", type=int, default=1)
    parser.add_argument("--stat-color", type=int, default=38)
    # Layout controls.  Margins are fractions of the full canvas.  A wider
    # right margin is selected automatically when values or their breakdown
    # are printed, but every value can be overridden from the command line.
    parser.add_argument("--canvas-width", type=int, default=1200)
    parser.add_argument("--canvas-height", type=int, default=None)
    parser.add_argument("--left-margin", type=float, default=0.20)
    parser.add_argument("--right-margin", type=float, default=None)
    parser.add_argument("--top-margin", type=float, default=0.08)
    parser.add_argument("--bottom-margin", type=float, default=0.15)
    parser.add_argument("--cms-label-offset", type=float, default=0.055,
                        help="Horizontal NDC distance between CMS and its extra label")
    parser.add_argument("--value-text-size", type=float, default=0.026)
    args = parser.parse_args()

    fixed_name = args.POI
    if args.translate is not None:
        with open(args.translate) as jsonfile:
            name_translate = json.load(jsonfile)
        fixed_name = name_translate.get(args.POI, fixed_name)
    x_title = args.x_title if args.x_title is not None else fixed_name

    scan_specs = load_scans(args)
    yvals = [1.0, 4.0]
    rows = []
    all_x = []
    for idx, spec in enumerate(scan_specs):
        full = build_scan("full_%d" % idx, args.POI, [spec["full"]], yvals, args.y_cut)
        best, err_hi, err_lo = full["val"]
        row = {
            "label": spec["label"],
            "full": full,
            "best": best,
            "err_hi": abs(err_hi),
            "err_lo": abs(err_lo),
            "stat": None,
            "syst_hi": None,
            "syst_lo": None,
        }
        all_x.extend([best - abs(err_lo), best + abs(err_hi)])

        if spec["stat"]:
            stat = build_scan("stat_%d" % idx, args.POI, [spec["stat"]], yvals, args.y_cut)
            stat_best, stat_hi, stat_lo = stat["val"]
            syst_hi, syst_lo = quad_subtract((abs(err_hi), abs(err_lo)), (abs(stat_hi), abs(stat_lo)), spec["label"])
            row.update({
                "stat": stat,
                "stat_best": stat_best,
                "stat_hi": abs(stat_hi),
                "stat_lo": abs(stat_lo),
                "syst_hi": syst_hi,
                "syst_lo": syst_lo,
            })
            all_x.extend([stat_best - abs(stat_lo), stat_best + abs(stat_hi)])
        rows.append(row)

    if args.band:
        center, lo, hi = [float(x) for x in args.band.split(":")]
        band = (center, abs(lo), abs(hi))
        all_x.extend([center - abs(lo), center + abs(hi)])
    elif args.band_from is not None:
        ref = rows[args.band_from - 1]
        band = (ref["best"], ref["err_lo"], ref["err_hi"])
    else:
        band = None

    xmin = args.x_min if args.x_min is not None else min(all_x) - 0.15 * (max(all_x) - min(all_x))
    xmax = args.x_max if args.x_max is not None else max(all_x) + 0.15 * (max(all_x) - min(all_x))
    if xmin == xmax:
        xmin -= 1.0
        xmax += 1.0

    n = len(rows)
    height = args.canvas_height if args.canvas_height is not None else max(520, 120 + 72 * n)
    if args.right_margin is not None:
        right_margin = args.right_margin
    elif args.show_values and args.show_breakdown:
        right_margin = 0.28
    elif args.show_values:
        right_margin = 0.20
    else:
        right_margin = 0.05

    canv = ROOT.TCanvas(args.output, args.output, args.canvas_width, height)
    pad = ROOT.TPad("pad", "pad", 0, 0, 1, 1)
    pad.SetLeftMargin(args.left_margin)
    pad.SetRightMargin(right_margin)
    pad.SetTopMargin(args.top_margin)
    pad.SetBottomMargin(args.bottom_margin)
    pad.SetTicks(1, 1)
    pad.Draw()
    pad.cd()

    frame = ROOT.TH2F("frame", "", 100, xmin, xmax, n, 0.0, float(n))
    frame.SetStats(0)
    frame.GetXaxis().SetTitle(x_title)
    frame.GetXaxis().SetTitleSize(0.060)
    frame.GetXaxis().SetLabelSize(0.050)
    # Five major intervals remain readable even for the narrow MH range.
    frame.GetXaxis().SetNdivisions(505)
    frame.GetYaxis().SetLabelSize(0.05)
    frame.GetYaxis().SetLabelOffset(0.010)
    frame.GetYaxis().SetTickLength(0)
    for i, row in enumerate(rows):
        frame.GetYaxis().SetBinLabel(n - i, row["label"])
    frame.Draw("AXIS")

    if band is not None:
        box = ROOT.TBox(band[0] - band[1], 0.0, band[0] + band[2], float(n))
        box.SetFillColor(ROOT.kGray)
        box.SetFillStyle(1001)
        box.Draw("same")
        frame.Draw("AXISSAME")

    if args.split_after is not None and 0 < args.split_after < n:
        ysep = n - args.split_after
        sep = ROOT.TLine(xmin, ysep, xmax, ysep)
        sep.SetLineWidth(2)
        sep.Draw("same")

    full_graph = ROOT.TGraphAsymmErrors(n)
    full_graph.SetMarkerStyle(20)
    full_graph.SetMarkerSize(1.2)
    full_graph.SetMarkerColor(args.marker_color)
    full_graph.SetLineColor(args.marker_color)
    full_graph.SetLineWidth(2)

    stat_graph = ROOT.TGraphAsymmErrors()
    stat_graph.SetMarkerStyle(20)
    stat_graph.SetMarkerSize(1.0)
    stat_graph.SetMarkerColor(args.stat_color)
    stat_graph.SetLineColor(args.stat_color)
    stat_graph.SetLineWidth(2)

    stat_i = 0
    for i, row in enumerate(rows):
        y = n - i - 0.5
        full_graph.SetPoint(i, row["best"], y)
        full_graph.SetPointError(i, row["err_lo"], row["err_hi"], 0.0, 0.0)
        if row["stat"] is not None:
            stat_graph.SetPoint(stat_i, row["stat_best"], y)
            stat_graph.SetPointError(stat_i, row["stat_lo"], row["stat_hi"], 0.0, 0.0)
            stat_i += 1

    full_graph.Draw("PZ SAME")
    if args.show_breakdown and stat_i > 0:
        stat_graph.Draw("PZ SAME")

    if args.show_values:
        text = ROOT.TLatex()
        text.SetTextFont(52)
        text.SetTextSize(args.value_text_size)
        text.SetTextAlign(12)
        x_text = xmax + 0.020 * (xmax - xmin)
        for i, row in enumerate(rows):
            y = n - i - 0.5
            value = format_unc((row["best"], row["err_hi"], -row["err_lo"]), args.precision)
            if args.show_breakdown and row["stat"] is not None:
                value += " = %.*f ^{+%.*f}_{-%.*f}(stat)  ^{+%.*f}_{-%.*f}(syst) GeV" % (
                    args.precision, row["best"],
                    args.precision, row["stat_hi"], args.precision, row["stat_lo"],
                    args.precision, row["syst_hi"], args.precision, row["syst_lo"],
                )
            text.DrawLatex(x_text, y, value)

    cms = ROOT.TLatex()
    cms.SetNDC()
    cms.SetTextAlign(11)
    cms.SetTextFont(61)
    cms.SetTextSize(0.060)
    header_y = 1.0 - 0.8 * args.top_margin
    cms.DrawLatex(args.left_margin, header_y, "CMS")
    if args.cms_label:
        cms.SetTextFont(52)
        cms.SetTextSize(0.050)
        cms.DrawLatex(args.left_margin + args.cms_label_offset, header_y, args.cms_label)
    cms.SetTextFont(42)
    cms.SetTextAlign(31)
    cms.SetTextSize(0.055)
    cms.DrawLatex(1.0 - right_margin, header_y, args.lumi_label)

    if args.show_breakdown and stat_i > 0:
        plot_right = 1.0 - right_margin
        plot_top = 1.0 - args.top_margin
        legend = ROOT.TLegend(plot_right - 0.13, plot_top - 0.10,
                              plot_right - 0.01, plot_top - 0.01, "", "NBNDC")
        legend.SetNColumns(1)
        legend.AddEntry(full_graph, "Total", "LP")
        legend.AddEntry(stat_graph, "Stat", "LP")
        legend.Draw()

    pad.RedrawAxis()
    canv.cd()
    canv.Print(args.output + ".pdf")
    canv.Print(args.output + ".png")

    outfile = ROOT.TFile(args.output + ".root", "RECREATE")
    outfile.WriteTObject(full_graph, "total")
    if stat_i > 0:
        outfile.WriteTObject(stat_graph, "stat")
    outfile.Close()

    for row in rows:
        msg = "%s: %s total +%.5g/-%.5g" % (row["label"], args.POI, row["err_hi"], row["err_lo"])
        if row["stat"] is not None:
            msg += ", stat +%.5g/-%.5g, syst +%.5g/-%.5g" % (
                row["stat_hi"], row["stat_lo"], row["syst_hi"], row["syst_lo"]
            )
        print(msg)


if __name__ == "__main__":
    main()
