import glob
import subprocess
import re
import os
import pandas as pd
import json
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import itertools
from numpy.random import default_rng
import numpy as np
from functools import reduce


def minnn_extract(r1, r2, minnn_output_base, output_suffix):
    cmd = f'minnn -Xmx25G extract \
    --pattern \'(R1:N{{*}})\^tggtatcaacgcagagt(UMI:NNNNtNNNNtNNNN)tc(R2:N{{*}})\' -f \
    --json-report "{minnn_output_base}{output_suffix}.json" \
    --input "{r1}" "{r2}" \
    --output "{minnn_output_base}{output_suffix}.mif"'

    process = subprocess.Popen([cmd], stdout=True, stderr=True, shell=True)
    process.wait()


def minnn_filter(minnn_output_base, filter_name, input_suffix, output_suffix):
    cmd = f'minnn -Xmx25G filter \'{filter_name}\' -f \
    --json-report "{minnn_output_base}{output_suffix}.json" \
    --input "{minnn_output_base}{input_suffix}.mif" \
    --output "{minnn_output_base}{output_suffix}.mif"'

    process = subprocess.Popen([cmd], stdout=True, stderr=True, shell=True)
    process.wait()


def minnn_sort(minnn_output_base, input_suffix, output_suffix):
    cmd = f'minnn -Xmx25G sort -f \
    --groups UMI \
    --input "{minnn_output_base}{input_suffix}.mif" \
    --output "{minnn_output_base}{output_suffix}.mif"'

    process = subprocess.Popen([cmd], stdout=True, stderr=True, shell=True)
    process.wait()


def minnn_correct(minnn_output_base, input_suffix, output_suffix):
    cmd = f'minnn -Xmx25G correct -f \
    --threads 10 \
    --json-report "{minnn_output_base}{output_suffix}.json" \
    --groups UMI \
    --input "{minnn_output_base}{input_suffix}.mif" \
    --output "{minnn_output_base}{output_suffix}.mif"'

    process = subprocess.Popen([cmd], stdout=True, stderr=True, shell=True)
    process.wait()


def minnn_consensus(minnn_output_base, input_suffix, output_suffix):
    cmd = f'minnn -Xmx25G consensus -f \
    --threads 10 \
    --json-report "{minnn_output_base}{output_suffix}.json" \
    --groups UMI \
    --input "{minnn_output_base}{input_suffix}.mif" \
    --output "{minnn_output_base}{output_suffix}.mif" > "{minnn_output_base}{output_suffix}.log" 2>&1'

    process = subprocess.Popen([cmd], stdout=True, stderr=True, shell=True)
    process.wait()


def minnn_mif2fastq(minnn_output_base, input_suffix, fastq_output_base):
    cmd = f'minnn -Xmx25G mif2fastq -f \
    --input "{minnn_output_base}{input_suffix}.mif" \
    --group R1="{fastq_output_base}_R1.fastq" \
    R2="{fastq_output_base}_R2.fastq"'

    process = subprocess.Popen([cmd], stdout=True, stderr=True, shell=True)
    process.wait()


def get_sample_name_from_string(filename):
    return re.sub("(?:.*/)?(.*?)(\_L00\d)?(?:_R1.*(?:.fastq|.gz))", r"\1", filename)


def minnn_run(r1, r2, minnn_output, fastq_output):
    sample_name = get_sample_name_from_string(r1)
    print(sample_name)
    minnn_output_base = minnn_output + sample_name
    fastq_output_base = fastq_output + sample_name

    minnn_extract(r1, r2, minnn_output_base, "_extract")
    minnn_filter(minnn_output_base, "NoWildcards(UMI)", "_extract", "_extracted_filter")
    minnn_sort(minnn_output_base, "_extracted_filter", "_extracted_filtered_sorted")
    minnn_correct(minnn_output_base, "_extracted_filtered_sorted", "_corrected")
    minnn_sort(minnn_output_base, "_corrected", "_corrected_sorted")
    minnn_consensus(minnn_output_base, "_corrected_sorted", "_consensus")
    minnn_filter(minnn_output_base, "MinConsensusReads = 2", "_consensus", "_consensus_filtered2")
    minnn_mif2fastq(minnn_output_base, "_consensus_filtered2", fastq_output_base)


def mixcr_run(species, material, Five_end, Three_end, adapters, r1, r2, output_path, analyze_param="", align_param="",
              assemble_param=""):
    samplename = get_sample_name_from_string(r1)
    output_base = output_path + samplename
    cmd = f'mixcr analyze amplicon -s {species} --starting-material {material} --5-end {Five_end} --3-end {Three_end} -f \
    -j \
    --adapters {adapters} {analyze_param}\
    --align "{align_param}" --assemble "{assemble_param}" \
    --report {output_base}.report {r1} {r2} {output_base}'

    process = subprocess.Popen([cmd], stdout=True, stderr=True, shell=True)
    process.wait()


def parse_anchor_points(data):
    anchor_points_regex = "^^(?:-?[0-9]*:){8}(?:-?[0-9]*):(?P<CDR3Begin>-?[0-9]*):(?P<V3Deletion>-?[0-9]*):(?P<VEnd>-?[0-9]*):(?P<DStart>-?[0-9]*):(?P<D5Deletion>-?[0-9]*):(?P<D3Deletion>-?[0-9]*):(?P<DEnd>-?[0-9]*):(?P<JStart>-?[0-9]*):(?P<J5Deletion>-?[0-9]*):(?P<CDR3End>-?[0-9]*):(?:-?[0-9]*:){2}(?:-?[0-9]*)$"
    data = pd.concat([data, data.refPoints.str.extract(anchor_points_regex, expand=True).apply(pd.to_numeric)], axis=1)
    return data


def read_mixcr_table(filename):
    result = pd.read_table(filename)
    if len(result) == 0:
        return None
    result = parse_anchor_points(result)
    result = result[["cloneCount", "cloneFraction", "nSeqCDR3", "aaSeqCDR3", "allVHitsWithScore",
                     "allDHitsWithScore", "allJHitsWithScore", "allCHitsWithScore", "VEnd", "DStart", "DEnd", "JStart"]]
    result = result.rename(columns={"cloneCount": "count",
                                    "cloneFraction": "frequency",
                                    "nSeqCDR3": "CDR3nt",
                                    "aaSeqCDR3": "CDR3aa",
                                    "allVHitsWithScore": "V",
                                    "allDHitsWithScore": "D",
                                    "allJHitsWithScore": "J",
                                    "allCHitsWithScore": "C"})
    result["V"] = result.V.str.replace("\*.*", "")
    result["D"] = result.D.fillna("N/A").str.replace("\*.*", "")
    result["J"] = result.J.str.replace("\*.*", "")
    result["C"] = result.C.str.replace("\*.*", "")
    result["N"] = result.apply(lambda row: count_n(row.VEnd, row.DStart, row.DEnd, row.JStart), axis=1)
    result["CDR3length"] = result.CDR3nt.str.len()
    result["count"] = result["count"].astype(int)
    return result


def only_productive(data):
    result = data.copy()
    result = result.loc[~result["CDR3aa"].str.contains("[\*,_]", regex=True)]
    result["frequency"] = result["count"] / result["count"].sum()
    return result


def get_key(name, corr_dict, default):
    for pattern, value in corr_dict.items():
        if pattern in name:
            return value
    return default


def get_feature(data, feature, weighted=True):
    if weighted and data.frequency.sum() != 0:
        return (data[feature] * data.frequency).sum() / data.frequency.sum()
    elif data.frequency.sum() != 0:
        return data[feature].mean()
    else:
        return 0


def count_n(VEnd, DBegin, DEnd, JBegin):
    if pd.notna(DBegin):
        if VEnd != DBegin:
            VD = DBegin - VEnd
        else:
            VD = 0
        if JBegin != DEnd:
            N = JBegin - DEnd
        else:
            DJ = 0
            N = VD + DJ
    else:
        if JBegin != VEnd:
            N = JBegin - VEnd
        else:
            return 0
    if N <= 0:
        return 0
    else:
        return N


def basic_analysis(mixcr_path, chain_dict, material_dict, full_clonesets_export_path, functional_clonesets_export_path,
                   minnn_path=""):
    general_samples_dict = {}

    # Create folders if needed
    create_folder(full_clonesets_export_path)
    create_folder(functional_clonesets_export_path)

    # Create list of samples based on .clns filees in mixcr folder
    samples = []
    for filename in glob.glob(mixcr_path + "*.vdjca"):
        samples.append(re.sub("(.*/)(.*)(\.vdjca)", r"\2", filename))

    # get chain based on chainDict
    for sample in samples:
        if "Undetermined" in sample:
            continue
        print(sample)
        chain = get_key(sample, chain_dict, "ALL")
        # load sample file
        data = read_mixcr_table(mixcr_path + sample + ".clonotypes." + chain + ".txt")
        if data is None:
            continue
        general_samples_dict[sample] = {}

        full_sample_path = full_clonesets_export_path + sample + ".txt"
        functional_sample_path = functional_clonesets_export_path + sample + ".txt"

        # save pretty file to the folder
        data.to_csv(full_sample_path, sep="\t", index=False)

        data_productive = only_productive(data)
        data_productive.to_csv(functional_sample_path, sep="\t", index=False)

        general_samples_dict[sample]["fullSamplePath"] = full_sample_path
        general_samples_dict[sample]["functionalSamplePath"] = functional_sample_path
        general_samples_dict[sample]["material"] = get_key(sample, material_dict, "RNA")
        general_samples_dict[sample]["productiveClonesNmbr"] = len(data_productive)
        general_samples_dict[sample]["productiveReadsNmbr"] = data_productive["count"].sum()
        general_samples_dict[sample]["meanCDR3"] = get_feature(data_productive, "CDR3length", weighted=True)
        general_samples_dict[sample]["meanN"] = get_feature(data_productive, "N", weighted=True)
        general_samples_dict[sample]["chain"] = chain

        if minnn_path != "":
            general_samples_dict[sample]["extract_report"] = json.load(open(minnn_path + sample + "_extract.json"))
            general_samples_dict[sample]["extracted_filter"] = json.load(
                open(minnn_path + sample + "_extracted_filter.json"))
            general_samples_dict[sample]["consensus_report"] = json.load(open(minnn_path + sample + "_consensus.json"))
            general_samples_dict[sample]["consensus_filter_report"] = json.load(
                open(minnn_path + sample + "_consensus_filtered2.json"))
        general_samples_dict[sample]["align_report"] = json.load(open(mixcr_path + sample + "_align.json"))
        general_samples_dict[sample]["assemble_report"] = json.load(open(mixcr_path + sample + "_assemble.json"))

    return general_samples_dict


def get_report(samples_dict, output_path, minnn=True):
    # Generating report
    report = pd.DataFrame()
    for sample, metadata in samples_dict.items():
        single = {"Sample_id": sample, "Starting Material": metadata["material"]}

        if minnn:
            single["Total reads"] = metadata["extract_report"]["totalReads"]
            single["Reads matched pattern"] = metadata["extract_report"]["matchedReads"]
            single["Reads passed 'NoWildcards' filter"] = metadata["extracted_filter"]["matchedReads"]
            single["Reads used in consensus"] = single["Reads passed 'NoWildcards' filter"] - \
                                                metadata["consensus_report"]["notUsedReadsCount"]
            single["Total consensuses"] = metadata["consensus_report"]["consensusReads"]
            single["Number of consensuses with overseq more then 2"] = metadata["align_report"]["totalReadsProcessed"]
        else:
            single["Total reads"] = metadata["align_report"]["totalReadsProcessed"]
        single["Aligned consensuses"] = metadata["align_report"]["aligned"]
        single["Number of consensuses in clonotypes"] = metadata["assemble_report"]["readsInClones"]
        single["Number of clonotypes"] = metadata["assemble_report"]["clones"]

        single["Mean weighted CDR3 length"] = metadata["meanCDR3"]
        single["Mean weighted insert size"] = metadata["meanN"]

        single["Number of productive clonotypes"] = metadata["productiveClonesNmbr"]
        single["Number of consensuses in clonotypes after filtration"] = metadata["productiveReadsNmbr"]

        report = report.append(single, ignore_index=True)

    columns = ["Sample_id", "Starting Material", "Total reads"]
    if minnn:
        columns.extend(("Reads matched pattern", "Reads passed 'NoWildcards' filter", "Reads used in consensus",
                        "Total consensuses", "Number of consensuses with overseq more then 2"))
    columns.extend(("Aligned consensuses", "Number of consensuses in clonotypes", "Number of clonotypes",
                    "Number of consensuses in clonotypes after filtration", "Number of productive clonotypes",
                    "Mean weighted CDR3 length", "Mean weighted insert size"))

    report = report[columns]
    if not minnn:
        report.columns = report.columns.str.replace("consensuses", "reads")
    report.sort_values("Sample_id").to_csv(output_path + "report.txt", sep="\t", index=False)
    return report


# Returns file1 count, file2 count, file12 count, file21 count, file1 div, file2 div, div12,
# freq1, freq2, freq12, freq21
def intersect_pair(data1, data2, by):
    if by == "nt":
        on_column = "CDR3nt"
    elif by == "aa":
        on_column = "CDR3aa"
    else:
        return ("Wrong intersect parameter")
    merge = pd.merge(data1, data2, on=[on_column, "V", "J", "C"], suffixes=('_1', '_2'), how="inner")

    return round(data1["count"].sum()), \
           round(data2["count"].sum()), \
           round(merge.count_1.sum()), \
           round(merge.count_2.sum()), \
           round(len(data1[on_column].unique())), \
           round(len(data2[on_column].unique())), \
           round(len(merge[on_column].unique())), \
           data1["frequency"].sum(), \
           data2["frequency"].sum(), \
           merge.frequency_1.sum(), \
           merge.frequency_2.sum()


def intersect(samples_dict, chain, by="nt", output_path="", functional=False):
    intersect_table = pd.DataFrame()
    for pair in list(itertools.combinations(samples_dict, 2)):
        row = {}
        if samples_dict[pair[0]]["chain"] != chain or samples_dict[pair[1]]["chain"] != chain:
            continue
        sample1 = get_sample_data(samples_dict, pair[0], functional)
        sample2 = get_sample_data(samples_dict, pair[1], functional)
        row["Sample_id1"] = pair[0]
        row["Sample_id2"] = pair[1]

        row["count1"], row["count2"], row["count12"], row["count21"], row["div1"], row["div2"], row["div12"], \
        row["freq1"], row["freq2"], row["freq12"], row["freq21"] = intersect_pair(sample1, sample2, by=by)
        intersect_table = intersect_table.append(row, ignore_index=True)

    if len(intersect_table) != 0:
        intersect_table = intersect_table[
            ["Sample_id1", "Sample_id2", "count1", "count2", "count12", "count21", "div1", "div2",
             "div12", "freq1", "freq2", "freq12", "freq21"]]
        intersect_table = intersect_table.sort_values(["Sample_id1", "Sample_id2"])
        if output_path != "":
            intersect_table.to_csv(output_path + "intersect" + chain + ".txt", sep="\t", index=False)
    return intersect_table


def vj_usage(samples_dict, segment, chain, functional=False, weighted=True, output_path="", plot=True):
    if not os.path.exists(output_path) and output_path != "":
        os.makedirs(output_path)

    sample = {}
    for sample_id, metadata in samples_dict.items():
        if metadata["chain"] != chain:
            continue
        data = get_sample_data(samples_dict, sample_id, functional)
        if weighted:
            sample[sample_id] = data.groupby(segment, axis=0)["count"].sum().apply(lambda x: x / data["count"].sum())
        else:
            sample[sample_id] = data.groupby(segment, axis=0)["count"].count().apply(
                lambda x: x / data["count"].count())

    result = pd.DataFrame(sample).fillna(0)
    if plot:
        plot = sns.clustermap(result, z_score=1, cmap="coolwarm", xticklabels=True, yticklabels=True)
    if output_path != "":
        if plot:
            plot.savefig(output_path + segment + ".usage.pdf")
        result.to_csv(output_path + segment + ".usage.txt", sep="\t")
    return result


def get_sample_data(samples_dict, sample_id, functional=False):
    if functional:
        return pd.read_table(samples_dict[sample_id]["functionalSamplePath"])
    return pd.read_table(samples_dict[sample_id]["fullSamplePath"])


def plot_samples_intersect(samples_dict, chain, functional=False, equalby=None, figzise=(80, 80),
                           output_path="", ylim=(-0.2, 40000), xlim=(-0.2, 40000)):
    if equalby is None:
        equalby = ['CDR3nt', 'V', 'J', "C"]
    valid_chain_dict = {}
    for sampleId, metadata in samples_dict.items():
        if metadata["chain"] != chain:
            continue
        valid_chain_dict[sampleId] = metadata

    nmb_of_samples = len(valid_chain_dict)
    samples_pair_list = list(itertools.combinations_with_replacement(sorted(valid_chain_dict), 2))
    fig, axes = plt.subplots(nrows=nmb_of_samples, ncols=nmb_of_samples, figsize=figzise, sharex=True, sharey=True)

    fig.tight_layout(pad=5.0)

    k = -1
    for i in reversed(range(nmb_of_samples)):
        for j in reversed(range(nmb_of_samples)):
            if i < j:
                axes[i, j].axis('off')
            else:
                k += 1
                sample1 = get_sample_data(valid_chain_dict, samples_pair_list[k][1], functional)
                sample2 = get_sample_data(valid_chain_dict, samples_pair_list[k][0], functional)
                sample1id = samples_pair_list[k][1]
                sample2id = samples_pair_list[k][0]

                merge = sample1[['count', 'CDR3nt', "CDR3aa", 'V', 'J', 'C']].merge(
                    sample2[['count', "CDR3aa", 'CDR3nt', 'V', 'J', 'C']],
                    how='outer', on=equalby,
                    suffixes=['_1' + sample1id, '_2' + sample2id]).fillna(0)
                sns.set(style="white", color_codes=True)
                plot = sns.regplot(ax=axes[i][j], x='count_1' + sample1id, y='count_2' + sample2id,
                                   data=merge, fit_reg=False, scatter_kws={"s": 100})

                plot.set(ylim=ylim, xlim=xlim, xscale="symlog", yscale="symlog")
                xlabel = sample1id
                ylabel = sample2id
                if j != 0:
                    ylabel = ""
                if i != nmb_of_samples - 1:
                    xlabel = ""
                axes[i][j].set_xticks([0, 1, 100, 10000])
                axes[i][j].set_xticklabels(['0', '1', '100', '10000'], fontsize=35)
                axes[i][j].set_yticks([0, 1, 100, 10000])
                axes[i][j].set_yticklabels(['0', '1', '100', '10000'], fontsize=35)

                axes[i][j].set_xlabel(xlabel, fontsize=40, rotation=45, verticalalignment='top',
                                      horizontalalignment='right')

                axes[i][j].set_ylabel(ylabel, fontsize=40, rotation=45, verticalalignment='top',
                                      horizontalalignment='right', y=1.0)
    if output_path != "":
        create_folder(output_path)
        fig.savefig(output_path + ".intersect.pdf", bbox_inches="tight")
        fig.savefig(output_path + ".intersect.png", bbox_inches="tight")


def create_folder(path):
    if path != "":
        output_folder = re.sub("(.*/)(.*)", r"\1", path)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)


def filter_samples_dict(samples_dict, patterns: list):
    new_dict = {}
    for sample_id, metadata in sorted(samples_dict.items()):
        if all(c in sample_id for c in patterns):
            new_dict[sample_id] = metadata
    return new_dict


def downsample(samples_dict, output_folder, x, functional=True):
    if output_folder is None:
        return "Please specify output folder"
    create_folder(output_folder)
    updated_samples_dict = {}
    if x < 0:
        return "X must be a positive number"
    for sample, metadata in samples_dict.items():
        data = get_sample_data(samples_dict, sample, functional)
        rng = default_rng()
        downsample_nmbr = x if x < data["count"].sum() else data["count"].sum()
        data["count"] = rng.multivariate_hypergeometric(data["count"].astype(np.int64), downsample_nmbr)
        data = data.loc[data["count"] != 0]
        output_path = output_folder + sample + "_" + str(downsample_nmbr) + ".txt"
        data.to_csv(output_path, sep="\t", index=False)
        updated_samples_dict[sample] = metadata
        updated_samples_dict[sample]["downsamplePath"] = output_path
    return updated_samples_dict


def merge_clonesets(samples_list, names_list, on="CDR3nt", value="count", how="outer", function="sum"):
    """Merge clonesets. Function takes a list of samples generated by basic_analysis and a list of labels for each sample.
    Indices in both list should correspond. By default merge_clonesets will merge clonesets on CDR3 nucleotide sequence
    and return a table with counts for each CDR3 sequence (rows) for every sample (column). Any column can be used as a
    merge on (ex. V , J, C, CDR3aa). Value can be set to any numeric column (ex. count, frequency, N, CDR3length) to
    demonstrate the value for a particular feature the clonesets were merged on.
    Function should be one of: sum, count or mean and will be applied to the value series in each group."""

    if len(samples_list) != len(names_list):
        return "List of samples and list of names must be the same length"
    if not any(function == func for func in ['sum', 'count', 'mean']):
        return "Function must be one of: sum, count or mean"
    samples = samples_list.copy()
    samples = [c.groupby(on).agg(function)[value] for c in samples]
    df_final = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how=how),
                      samples).fillna(0)
    df_final.columns = names_list
    return df_final.reset_index()
