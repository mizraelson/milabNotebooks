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


def minnn_extract(R1, R2, minnn_output_base, output_suffix):
    cmd = f'minnn -Xmx25G extract \
    --pattern \'(R1:N{{*}})\^tggtatcaacgcagagt(UMI:NNNNtNNNNtNNNN)tc(R2:N{{*}})\' -f \
    --json-report "{minnn_output_base}{output_suffix}.json" \
    --input "{R1}" "{R2}" \
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


def minnn_run(R1, R2, minnn_output, fastq_output):
    samplename = re.sub("(.*/)(.*)(_R[1,2].*\.gz)", r"\2", R1)
    print(samplename)
    minnn_output_base = minnn_output + samplename
    fastq_output_base = fastq_output + samplename

    minnn_extract(R1, R2, minnn_output_base, "_extract")
    minnn_filter(minnn_output_base, "NoWildcards(UMI)", "_extract", "_extracted_filter")
    minnn_sort(minnn_output_base, "_extracted_filter", "_extracted_filtered_sorted")
    minnn_correct(minnn_output_base, "_extracted_filtered_sorted", "_corrected")
    minnn_sort(minnn_output_base, "_corrected", "_corrected_sorted")
    minnn_consensus(minnn_output_base, "_corrected_sorted", "_consensus")
    minnn_filter(minnn_output_base, "MinConsensusReads = 2", "_consensus", "_consensus_filtered2")
    minnn_mif2fastq(minnn_output_base, "_consensus_filtered2", fastq_output_base)


def mixcr_run(species, material, Five_end, Three_end, adapters, R1, R2, output_path):
    samplename = re.sub("(?:.*/)?(.*?)(\_L00\d)?(?:_R1.*(?:.fastq|.gz))", r"\1", R1)
    output_base = output_path + samplename
    cmd = f'mixcr analyze amplicon -s {species} --starting-material {material} --5-end {Five_end} --3-end {Three_end} -f \
    --adapters {adapters} \
    --align "-j {output_base}_align.json" --assemble "-j {output_base}_assemble.json" \
    --report {output_base}.report {R1} {R2} {output_base}'

    process = subprocess.Popen([cmd], stdout=True, stderr=True, shell=True)
    process.wait()


def parse_anchorPoints(data):
    anchorPointsRegex = "^^(?:-?[0-9]*:){8}(?:-?[0-9]*):(?P<CDR3Begin>-?[0-9]*):(?P<V3Deletion>-?[0-9]*):(?P<VEnd>-?[0-9]*):(?P<DStart>-?[0-9]*):(?P<D5Deletion>-?[0-9]*):(?P<D3Deletion>-?[0-9]*):(?P<DEnd>-?[0-9]*):(?P<JStart>-?[0-9]*):(?P<J5Deletion>-?[0-9]*):(?P<CDR3End>-?[0-9]*):(?:-?[0-9]*:){2}(?:-?[0-9]*)$"
    data = pd.concat([data, data.refPoints.str.extract(anchorPointsRegex, expand=True).apply(pd.to_numeric)], axis=1)
    return data


def read_mixcr_table(filename):
    result = pd.read_table(filename)
    result = parse_anchorPoints(result)
    result = result[["cloneCount", "cloneFraction", "nSeqCDR3", "aaSeqCDR3", "allVHitsWithScore", \
                     "allDHitsWithScore", "allJHitsWithScore", "VEnd", "DStart", "DEnd", "JStart"]]
    result = result.rename(columns={"cloneCount": "count", \
                                    "cloneFraction": "frequency", \
                                    "nSeqCDR3": "CDR3nt", \
                                    "aaSeqCDR3": "CDR3aa", \
                                    "allVHitsWithScore": "V", \
                                    "allDHitsWithScore": "D", \
                                    "allJHitsWithScore": "J"})
    result["V"] = result.V.str.replace("\*.*", "")
    result["D"] = result.D.fillna("N/A").str.replace("\*.*", "")
    result["J"] = result.J.str.replace("\*.*", "")
    result["N"] = result.apply(lambda row: countN(row.VEnd, row.DStart, row.DEnd, row.JStart), axis=1)
    result["CDR3length"] = result.CDR3nt.str.len()
    result["count"] = result["count"].astype(int)
    return result


def only_productive(data):
    result = data.copy()
    result = result.loc[~result["CDR3aa"].str.contains("[\*,_]", regex=True)]
    result["frequency"] = result["count"] / result["count"].sum()
    return result


def get_key(name, corrDict, default):
    for pattern, value in corrDict.items():
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


def countN(VEnd, DBegin, DEnd, JBegin):
    if pd.notna(DBegin):
        if (VEnd != DBegin):
            VD = DBegin - VEnd
        else:
            VD = 0
        if (JBegin != DEnd):
            DJ = JBegin - DEnd
        else:
            DJ = 0
        return VD + DJ
    else:
        if (JBegin != VEnd):
            return JBegin - VEnd
        else:
            return 0;


def basicAnalisis(mixcr_path, chainDict, materialDict, fullClonesetsExportPath, functionalClonesetsExportPath,
                  minnn_path=""):
    generalSamplesDict = {}

    # Create folders if needed
    if not os.path.exists(fullClonesetsExportPath) and fullClonesetsExportPath != "":
        os.makedirs(fullClonesetsExportPath)
    if not os.path.exists(functionalClonesetsExportPath) and functionalClonesetsExportPath != "":
        os.makedirs(functionalClonesetsExportPath)

    # Create list of samples based on .clns filees in mixcr folder
    samples = []
    for filename in glob.glob(mixcr_path + "*.cln*"):
        samples.append(re.sub("(.*/)(.*)(\.cln[a,s])", r"\2", filename))

    # get chain based on chainDict
    for sample in samples:
        if "Undetermined" in sample:
            continue
        generalSamplesDict[sample] = {}

        chain = get_key(sample, chainDict, "ALL")
        # load sample file
        data = read_mixcr_table(mixcr_path + sample + ".clonotypes." + chain + ".txt")
        fullSamplePath = fullClonesetsExportPath + sample + ".txt"

        # save pretty file to the folder
        data.to_csv(fullSamplePath, sep="\t", index=False)

        dataProductive = only_productive(data)

        functionalSamplePath = functionalClonesetsExportPath + sample + ".txt"

        dataProductive.to_csv(functionalSamplePath, sep="\t", index=False)

        generalSamplesDict[sample]["fullSamplePath"] = fullSamplePath
        generalSamplesDict[sample]["functionalSamplePath"] = functionalSamplePath
        generalSamplesDict[sample]["material"] = get_key(sample, materialDict, "RNA")
        generalSamplesDict[sample]["productiveClonesNmbr"] = len(dataProductive)
        generalSamplesDict[sample]["productiveReadsNmbr"] = dataProductive["count"].sum()
        generalSamplesDict[sample]["meanCDR3"] = get_feature(dataProductive, "CDR3length", weighted=True)
        generalSamplesDict[sample]["meanN"] = get_feature(dataProductive, "N", weighted=True)
        generalSamplesDict[sample]["chain"] = chain

        if minnn_path != "":
            generalSamplesDict[sample]["extract_report"] = json.load(open(minnn_path + sample + "_extract.json"))
            generalSamplesDict[sample]["extracted_filter"] = json.load(
                open(minnn_path + sample + "_extracted_filter.json"))
            generalSamplesDict[sample]["consensus_report"] = json.load(open(minnn_path + sample + "_consensus.json"))
            generalSamplesDict[sample]["consensus_filter_report"] = json.load(
                open(minnn_path + sample + "_consensus_filtered2.json"))
        generalSamplesDict[sample]["align_report"] = json.load(open(mixcr_path + sample + "_align.json"))
        generalSamplesDict[sample]["assemble_report"] = json.load(open(mixcr_path + sample + "_assemble.json"))

    return generalSamplesDict


def get_report(samplesDict, output_path, minnn=True):
    # Generating report
    report = pd.DataFrame()
    for sample, metadata in samplesDict.items():
        single = {}
        single["Sample_id"] = sample
        single["Starting Material"] = metadata["material"]

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
        columns.extend(("Reads matched pattern", "Reads passed 'NoWildcards' filter", "Reads used in consensus", \
                        "Total consensuses", "Number of consensuses with overseq more then 2"))
    columns.extend(("Aligned consensuses", "Number of consensuses in clonotypes", "Number of clonotypes", \
                    "Number of consensuses in clonotypes after filtration", "Number of productive clonotypes", \
                    "Mean weighted CDR3 length", "Mean weighted insert size"))

    report = report[columns]
    if not minnn:
        report.columns = report.columns.str.replace("consensuses", "reads")
    report.sort_values("Sample_id").to_csv(output_path + "report.txt", sep="\t", index=False)
    return report


# Returns file1 count, file2 count, file12 count, file21 count, file1 div, file2 div, div12,
# freq1, freq2, freq12, freq21
def intersectPair(data1, data2, by):
    if by == "nt":
        onColumn = "CDR3nt"
    elif by == "aa":
        onColumn = "CDR3aa"
    else:
        return ("Wrong intersect parameter")
    merge = pd.merge(data1, data2, on=[onColumn, "V", "J"], suffixes=('_1', '_2'), how="inner")

    return round(data1["count"].sum()), \
           round(data2["count"].sum()), \
           round(merge.count_1.sum()), \
           round(merge.count_2.sum()), \
           round(len(data1[onColumn].unique())), \
           round(len(data2[onColumn].unique())), \
           round(len(merge[onColumn].unique())), \
           data1["frequency"].sum(), \
           data2["frequency"].sum(), \
           merge.frequency_1.sum(), \
           merge.frequency_2.sum()


def intersect(samplesDict, chain, by="nt", output_path="", functional=False):
    intersectTable = pd.DataFrame()
    for pair in list(itertools.combinations(samplesDict, 2)):
        row = {}
        if samplesDict[pair[0]]["chain"] != chain or samplesDict[pair[1]]["chain"] != chain:
            continue
        sample1 = getSampleTable(samplesDict, pair[0], functional)
        sample2 = getSampleTable(samplesDict, pair[1], functional)
        row["Sample_id1"] = pair[0]
        row["Sample_id2"] = pair[1]

        row["count1"], row["count2"], row["count12"], row["count21"], row["div1"], row["div2"], row["div12"], \
        row["freq1"], row["freq2"], row["freq12"], row["freq21"] = intersectPair(sample1, sample2, by=by)
        intersectTable = intersectTable.append(row, ignore_index=True)

    if len(intersectTable) != 0:
        intersectTable = intersectTable[
            ["Sample_id1", "Sample_id2", "count1", "count2", "count12", "count21", "div1", "div2",
             "div12", "freq1", "freq2", "freq12", "freq21"]]
        intersectTable = intersectTable.sort_values(["Sample_id1", "Sample_id2"])
        if output_path != "":
            intersectTable.to_csv(output_path + "intersect" + chain + ".txt", sep="\t", index=False)
    return intersectTable


def VJusage(samples_dict, segment, chain, functional=False, weighted=True, outputPath="", plot=True):
    if not os.path.exists(outputPath) and outputPath != "":
        os.makedirs(outputPath)

    sample = {}
    for sample_id, metadata in samples_dict.items():
        if metadata["chain"] != chain:
            continue
        data = getSampleTable(samples_dict, sample_id, functional)
        if weighted:
            sample[sample_id] = data.groupby(segment, axis=0)["count"].sum().apply(lambda x: x / data["count"].sum())
        else:
            sample[sample_id] = data.groupby(segment, axis=0)["count"].count().apply(
                lambda x: x / data["count"].count())

    result = pd.DataFrame(sample).fillna(0)
    if plot:
        plot = sns.clustermap(result, z_score=1, cmap="coolwarm", xticklabels=True, yticklabels=True)
    if outputPath != "":
        if plot:
            plot.savefig(outputPath + segment + ".usage.pdf")
        result.to_csv(outputPath + segment + ".usage.txt", sep="\t")
    return result


def getSampleTable(sampleDict, sampleid, functional=False):
    if functional:
        return pd.read_table(sampleDict[sampleid]["functionalSamplePath"])
    return pd.read_table(sampleDict[sampleid]["fullSamplePath"])


def plotIntersectCorrelations(samplesDict, chain, name_pattern: list = None, functional=False, equalby=None,
                              figzise=(80, 80), output_path="", ylim=(-0.2, 40000), xlim=(-0.2, 40000)):
    if name_pattern is None:
        name_pattern = [chain]
    else:
        name_pattern.append(chain)
    if equalby is None:
        equalby = ['CDR3nt', 'V', 'J']

    processed_dict = samplesWithdraw(samplesDict, name_pattern)

    # for sampleId, metadata in sorted(samplesDict.items()):
    #     if metadata["chain"] != chain:
    #         continue
    #     processed_dict[sampleId] = metadata

    nmbOfSamples = len(processed_dict)
    samplesPairList = sorted(itertools.combinations_with_replacement(processed_dict, 2),
                             key=lambda element: (element[0], element[1]), reverse=True)
    fig, axes = plt.subplots(nrows=nmbOfSamples, ncols=nmbOfSamples, figsize=figzise, sharex=True, sharey=True)

    fig.tight_layout(pad=7.0)

    k = -1
    for i in range(nmbOfSamples):
        for j in range(nmbOfSamples):
            if i < j:
                axes[i, j].axis('off')
            else:
                k += 1
                sample1 = getSampleTable(processed_dict, samplesPairList[k][0], functional)
                sample2 = getSampleTable(processed_dict, samplesPairList[k][1], functional)
                sample1id = samplesPairList[k][0]
                sample2id = samplesPairList[k][1]

                merge = sample1[['count', 'CDR3nt', "CDR3aa", 'V', 'J']].merge(
                    sample2[['count', "CDR3aa", 'CDR3nt', 'V', 'J']],
                    how='outer', on=equalby,
                    suffixes=['_1' + sample1id, '_2' + sample2id]).fillna(0)
                sns.set(style="white", color_codes=True)
                plot = sns.regplot(ax=axes[i][j], x='count_2' + sample2id, y='count_1' + sample1id,
                                   data=merge, fit_reg=False, scatter_kws={"s": 100})

                plot.set(ylim=ylim, xlim=xlim, xscale="symlog", yscale="symlog")
                axes[i][j].set_xticks([0, 1, 100, 10000])
                axes[i][j].set_xticklabels(['0', '1', '100', '10000'], fontsize=35)
                axes[i][j].set_yticks([0, 1, 100, 10000])
                axes[i][j].set_yticklabels(['0', '1', '100', '10000'], fontsize=35)
                #             axes[i][j].set_xlabel("")
                #             axes[i][j].set_xlabel("")

                axes[i][j].set_xlabel(sample2id, fontsize=40)

                axes[i][j].set_ylabel(sample1id, fontsize=40)
        createFolder(output_path)
        fig.savefig(output_path + ".intersect.pdf", bbox_inches='tight')
        fig.savefig(output_path + ".intersect.png", bbox_inches='tight')


def createFolder(path):
    if path != "":
        output_folder = re.sub("(.*/)(.*)", r"\1", path)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)


def samplesWithdraw(samples_dict, patterns: list):
    new_dict = {}
    for sample_id, metadata in sorted(samples_dict.items()):
        for pattern in patterns:
            if pattern in sample_id:
                new_dict[sample_id] = metadata
    return new_dict
