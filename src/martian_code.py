from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import math

# ================= PATHS / DIRECTORIES ONLY =================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

#---------------- TASK 1: Nucleotides ----------------

def read_fasta(filepath: str | Path) -> str:
    """Reads a FASTA file and returns the sequence as a string."""
    sequence_parts = []
    filepath = Path(filepath)
    with open(filepath, 'r', encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('>'):
                continue
            sequence_parts.append(line)
    sequence = ''.join(sequence_parts)

    if not sequence:
        raise ValueError(f"No sequence data found in {filepath}")

    return sequence



def count_symbols(sequence: str) -> Counter:
    """Counts the occurrences of each symbol in the sequence."""
    return Counter(sequence)


def sorted_counts_table(counts: Counter) -> list[tuple[str, int, float]]:
    """Returns list of (symbol, count, percent) sorted by count descending then symbol."""
    total = sum(counts.values())
    table = []
    for sym, c in counts.most_common():
        pct = (c / total) * 100 if total else 0.0
        table.append((sym, c, pct))
    return table



def gc_content(sequence: str) -> float:
    """Compute GC content as a fraction of symbols that are G or C."""
    if not sequence:
        return 0.0
    seq = sequence.upper()
    gc = seq.count('G') + seq.count('C')
    return gc / len(seq)



def kmer_counts(sequence: str, k: int) -> Counter:
    """Counts k-mers in a sequence."""
    if k <= 0:
        raise ValueError("k must be a positive integer")
    if len(sequence) < k:
        return Counter()
    return Counter(sequence[i:i+k] for i in range(len(sequence) - k + 1))



def ensure_results_dir() -> Path:
    out_dir = RESULTS_DIR
    out_dir.mkdir(exist_ok=True)
    return out_dir



def save_bar_plot(counts: Counter, title: str, filename: str) -> None:
    """Saves a bar plot of symbol frequencies."""
    out_dir = ensure_results_dir()
    symbols = list(counts.keys())
    values = [counts[s] for s in symbols]

    plt.figure()
    plt.bar(symbols, values)
    plt.title(title)
    plt.xlabel("Symbol")
    plt.ylabel("Count")
    plt.savefig(out_dir / filename, dpi=200)
    plt.close()



def run_task1(dna: str, rna: str, protein: str) -> None:
    out_dir = ensure_results_dir()

    dna_counts = count_symbols(dna)
    rna_counts = count_symbols(rna)
    protein_counts = count_symbols(protein)

    #Text summary for report
    lines = []
    lines.append("=== TASK 1: NUCLEOTIDES / SYMBOL ANALYSIS ===\n")

    for name, seq, counts in [
        ("DNA (gene_a)", dna, dna_counts),
        ("RNA (rna_a)", rna, rna_counts),
        ("Protein (protein_a)", protein, protein_counts),
    ]:
        lines.append(f"[{name}]")
        lines.append(f"Length: {len(seq)}")
        lines.append(f"Unique symbols ({len(counts)}): {', '.join(sorted(counts.keys()))}")

        #GC content only makes biological sense for DNA/RNA
        if name.startswith("DNA") or name.startswith("RNA"):
            gc = gc_content(seq)
            lines.append(f"GC content: {gc:.4f} ({gc*100:.2f}%)")

        lines.append("Frequency table (symbol, count, %):")

        for sym, c, pct in sorted_counts_table(counts):
            lines.append(f"  {sym:>2}  {c:>6}  {pct:>6.2f}%")

        # ---- Extended pattern analysis (kmers) ----
        if name.startswith("DNA") or name.startswith("RNA"):
            lines.append("Extended pattern analysis (k-mers):")
            for k in (2,3):
                kc = kmer_counts(seq, k)
                top  = kc.most_common(10)
                lines.append(f" Top {len(top)} {k}-mers:")
                for mer, c in top:
                    lines.append(f"   {mer} : {c}")


        lines.append("") #blank line between sections


    #Write summary to file
    summary_path = out_dir / "task1_summary.txt"
    summary_path.write_text('\n'.join(lines), encoding="utf-8")

    #Save plots
    save_bar_plot(dna_counts, "DNA Symbol Frequencies (gene_a)", "task1_dna_symbols.png")
    save_bar_plot(rna_counts, "RNA Symbol Frequencies (rna_a)", "task1_rna_symbols.png")
    save_bar_plot(protein_counts, "Protein Symbol Frequencies (protein_a)", "task1_protein_symbols.png")

    print(f"[Task 1] Wrote: {summary_path}")
    print(f"[Task 1] Plots saved to: {out_dir}")


#---------------- TASK 2: Transcription Key ----------------

def infer_transcription_key( dna: str, rna: str) -> tuple[int, dict[str, set[str]]]:
    """Infer the DNA -> RNA transcription mapping by testing different alignments.
    Returns the best alignment offset and the corresponding mapping."""

    max_offset = len(dna) - len(rna)
    if max_offset < 0:
        raise ValueError("RNA sequence is longer than DNA sequence; cannot align.")

    best_offset = 0
    best_mapping: dict[str, set[str]] = {}
    best_score = float("inf")

    for offset in range(max_offset + 1):
        mapping: dict[str, set[str]] = {}

        for i in range(len(rna)):
            d = dna[offset + i]
            r = rna[i]

            mapping.setdefault(d, set()).add(r)

        #Ambiguity Score: lower is better.
        ambiguity = sum(len(v) -1 for v in mapping.values())
        penalty = 0

        #Enforce known rule: T -> U.
        if "T" not in mapping or mapping["T"] != {"U"}:
            penalty += 1000

        score = ambiguity + penalty

        if score < best_score:
            best_score = score
            best_offset = offset
            best_mapping = mapping

        if score == 0:
            break #Cannot do better than perfect mapping.


    return best_offset, best_mapping



def run_task2(dna: str, rna: str) -> None:
    out_dir = ensure_results_dir()

    offset, mapping = infer_transcription_key(dna, rna)

    #Write summary to file
    lines = []
    lines.append("=== TASK 2: TRANSCRIPTION KEY ===\n")
    lines.append("DNA -> RNA base-pairing rules inferred from gene_a / rna_a\n")
    lines.append(f"Best alignment offset in DNA: {offset}\n")
        
    conflicts = False

    for dna_base in sorted(mapping.keys()):
        rna_bases = sorted(mapping[dna_base])
        lines.append(f"  {dna_base} -> {', '.join(rna_bases)}")

        if len(rna_bases) > 1:
            conflicts = True

    if conflicts:
        lines.append("\n[WARNING] At least one DNA symbol maps to multiple RNA symbols.")
        lines.append("This suggests ambiguity in the transcription process or alignment.")
    else:
        lines.append("\nALL DNA symbols map consistently to a single RNA symbol.")
        

    #Write to file
    out_path = out_dir / "task2_transcription_key.txt"
    out_path.write_text('\n'.join(lines), encoding="utf-8")

    print(f"[Task 2] Transcription key written to: {out_path}")



#---------------- TASK 3: Codon Length ----------------


def infer_codon_length(rna: str, protein: str, max_k: int = 12)  -> dict:
    """Try to infer the codon length k by checking which k (and reading frame) makes the
    number of codons match protein length (optionally +1 for a terminal stop codon). Also
    reports the theoretical minimum k based on alphabet sizes."""

    rna = rna.strip()
    protein = protein.strip()

    rna_symbols = set(rna)
    aa_symbols = set(protein)

    #Theoretical minimum based on information capacity
    #Need at least (#AA symbols + 1 stop) codons available
    needed = len(aa_symbols) + 1
    base = len(rna_symbols)
    k_min_theory = math.ceil(math.log(needed, base)) if base > 1 else None

    results = {
        "rna_length": len(rna),
        "protein_length": len(protein),
        "rna_alphabet_size": len(rna_symbols),
        "protein_alphabet_size": len(aa_symbols),
        "theoretical_min_k": k_min_theory,
        "candidates": []
    }

    #Empirical candidates: check k and frame offsets
    for k in range(1, max_k + 1):
        for frame in range(k):
            codon_count = (len(rna) - frame) // k
            remainder = (len(rna) - frame) % k

            if codon_count == len(protein) or codon_count == len(protein) + 1:
                results["candidates"].append({
                    "k": k,
                    "frame": frame,
                    "codon_count": codon_count,
                    "remainder": remainder,
                    "matches": "exact" if codon_count == len(protein) else "protein+1 (possible stop)"
                })

    return results



def run_task3(rna: str, protein: str) -> None:
    out_dir = ensure_results_dir()
    results = infer_codon_length(rna, protein)

    #Write summary to file
    lines = []
    lines.append("=== TASK 3: CODON LENGTH INFERENCE ===\n")

    lines.append(f"RNA length: {results['rna_length']}")
    lines.append(f"Protein length: {results['protein_length']}")
    lines.append(f"RNA alphabet size: {results['rna_alphabet_size']}")
    lines.append(f"Protein alphabet size: {results['protein_alphabet_size']}")

    lines.append(f"Theoretical minimum codon length (k): {results['theoretical_min_k']}\n")

    if results["candidates"]:
        lines.append("Empirical candidates found:")
        for c in results["candidates"]:
            lines.append(
                f"  k={c['k']}, frame={c['frame']}, "
                f"codons={c['codon_count']}, remainder={c['remainder']} "
                f"({c['matches']})"
            )
    else:
        lines.append("No fixed-length codon size and reading frame were found that "
        "allow the RNA sequence to translate directly into the given protein sequence.")

    lines.append("\nConclusion: \n"
    "The codon length must be at least 3 based on the RNA and protein alphabet sizes. "
    "However, the provided RNA and protein sequences do not correspond via a simple contiguous "
    "fixed length translation.")

    #Write to file
    out_path = out_dir / "task3_codon_length_inference.txt"
    out_path.write_text('\n'.join(lines), encoding="utf-8")

    print(f"[Task 3] Codon length inference written to: {out_path}")



#---------------- TASK 4: Codon Table ----------------


def chunk_codons(rna: str, k: int = 3, frame: int = 0) -> list[str]:
    """Split RNA into codons of length k starting at a given frame offset."""
    rna = rna.strip()
    codons = []
    for i in range(frame, len(rna) - k + 1, k):
        codons.append(rna[i:i+k]) 
    return codons


def infer_codon_table(rna: str, protein: str, k: int = 3, frame: int = 0) -> tuple[dict[str, set[str]], int]:
    """Infer codon -> amino acid mapping for a given frame by pairing codons with protein symbols.
    Returns (mapping, conflicts_count)"""

    codons = chunk_codons(rna, k=k, frame=frame)
    n = min(len(codons), len(protein))

    mapping: dict[str, set[str]] = {}
    for i in range(n):
        codon = codons[i]
        aa = protein[i]
        mapping.setdefault(codon, set()).add(aa)

    conflicts = sum(1 for aas in mapping.values() if len(aas) > 1)
    return mapping, conflicts


def choose_best_frame_for_codon_table(rna: str, protein: str, k: int = 3) -> tuple[int, dict[str, set[str]], int]:
    """Try frames 0..k-1 and select the frame with the fewest conflicts.
    Returns (best_frame, best_mapping, best_conflicts)"""

    best_frame = 0
    best_mapping: dict[str, set[str]] = {}
    best_conflicts = float("inf")

    for frame in range(k):
        mapping, conflicts = infer_codon_table(rna, protein, k=k, frame=frame)

        if conflicts < best_conflicts:
            best_conflicts = conflicts
            best_frame = frame
            best_mapping = mapping

        if conflicts == 0:
            break #Cannot do better than perfect mapping.

    return best_frame, best_mapping, int(best_conflicts)



def run_task4(rna: str, protein: str, k: int = 3) -> None:
    out_dir = ensure_results_dir()

    best_frame, mapping, conflicts = choose_best_frame_for_codon_table(rna, protein, k=k)

    #Write summary to file
    lines = []
    lines.append("=== TASK 4: CODON TABLE INFERENCE ===\n")
    lines.append(f"Assumed codon length (k): {k}")
    lines.append(f"Selected reading frame: {best_frame}")
    lines.append(f"Number of codons mapped: {len(mapping)}")
    lines.append(f"Conflicting/Ambiguous codons (codon -> multiple amino acids): {conflicts}\n")

    #Splitting the mapping into unambiguous and ambiguous codons
    unambiguous = {c: next(iter(aas)) for c, aas in mapping.items() if len(aas) == 1}
    ambiguous = {c: sorted(aas) for c, aas in mapping.items() if len(aas) > 1}

    total_possible = len(set(rna)) ** k
    coverage = (len(mapping) / total_possible) * 100 if total_possible else 0.0

    lines.append(f"Total possible codons (|alphabet|^k): {total_possible}")
    lines.append(f"Observed unique codons: {len(mapping)} ({coverage:.2f}% coverage)")
    lines.append(f"Unambiguous codons: {len(unambiguous)}\n")

    lines.append("Unambiguous codons (codon -> amino acid):")
    for codon in sorted(unambiguous.keys()):
        lines.append(f"  {codon} -> {unambiguous[codon]}")

    lines.append("\nAmbiguous codons (codon -> possible amino acids):")
    for codon in sorted(ambiguous.keys()):
        lines.append(f"  {codon} -> {', '.join(ambiguous[codon])}")



    if conflicts > 0:
        lines.append(
            "\nNote: conflicts indicate that the RNA/protein sequences may not align as "
            "a simple contiguous translation, or that additional rules (e.g., stop codons, " 
            "splicing, or alternative frames) may apply."
        )

    else:
        lines.append(
            "\nAll observed codons mapped consistently to a single amino acid."
        )


    #Write to file
    out_path = out_dir / "task4_codon_table.txt"
    out_path.write_text('\n'.join(lines), encoding="utf-8")

    print(f"[Task 4] Codon table written to: {out_path}")




#---------------- TASK 5: Transcription & Translation ----------------


def transcribe_dna_to_rna(dna: str, key: dict[str, str]) -> tuple[str, int]:
    """Transcribe DNA -> RNA using a substitution key. Skips 'N' bases."""
    rna_out = []
    skipped = 0
    for base in dna:
        if base == 'N':
            skipped += 1
            continue
        if base not in key:
            raise ValueError(f"DNA base '{base}' not in transcription key.")
        rna_out.append(key[base])
    return ''.join(rna_out), skipped


def make_translation_table(mapping: dict[str, set[str]]) -> dict[str, str]:
    """Keep only unambiguous codons (codon -> single amino acid)."""
    table: dict[str, str] = {}
    for codon, aas in mapping.items():
        if len(aas) == 1:
            table[codon] = next(iter(aas))

    return table


def translate_rna(rna: str, table: dict[str, str], k: int = 3, frame: int = 0) -> tuple[str, int, int]:
    """Translate RNA into protein. Unknown/ambiguous codons -> '?'.
    Returns (protein, codons_used, unknown_codons_count)"""

    codons = chunk_codons(rna, k=k, frame=frame)
    out = []
    unknown = 0

    for codon in codons:
        aa = table.get(codon, '?')
        if aa == '?':
            unknown += 1
        out.append(aa)
    return ''.join(out), len(codons), unknown


def compare_proteins(pred:str, true: str) -> dict[str, float | int]:
    """Compare predicted protein to true protein over the overlap"""
    n = min(len(pred), len(true))
    matches = mismatches = unknowns = comparable = 0

    for i in range(n):
        p = pred[i]
        t = true[i]
        if p == '?':
            unknowns += 1
            continue
        comparable += 1
        if p == t:
            matches += 1
        else:
            mismatches += 1

    identity = (matches / comparable * 100) if comparable else 0.0
    return {
        "overlap": n,
        "pred_len": len(pred),
        "true_len": len(true),
        "matches": matches,
        "mismatches": mismatches,
        "unknowns": unknowns,
        "comparable": comparable,
        "identity_percent": identity,
    }


def run_task5(dna: str, rna_given: str, protein_given: str, k: int = 3) -> None:
    out_dir = ensure_results_dir()

    # Task 2 transcription key (from your results)
    transcription_key = {
        "A": "Z",
        "B": "Y",
        "C": "X",
        "T": "U",
        # N is skipped intentionally
    }

    #Task 4 logic
    best_frame, codon_mapping, conflicts = choose_best_frame_for_codon_table(rna_given, protein_given, k=k)
    translation_table = make_translation_table(codon_mapping)

    #Transcribe DNA -> RNA, RNA -> protein prediction, compare predicted vs provided
    rna_from_dna, skipped_n = transcribe_dna_to_rna(dna, transcription_key)
    pred_protein, codons_used, unknown_codons = translate_rna(rna_from_dna, translation_table, k=k, frame=best_frame)
    stats = compare_proteins(pred_protein, protein_given)

    #Write to file
    lines = []
    lines.append("=== TASK 5: TRANSCRIPTION AND TRANSLATION ===\n")

    lines.append("Inputs:")
    lines.append(f"  DNA length: {len(dna)}")
    lines.append(f"  RNA (given) length: {len(rna_given)}")
    lines.append(f"  Protein (given) length: {len(protein_given)}\n")

    lines.append("Transcription (DNA -> RNA):")
    lines.append("  Rule: A->Z, B->Y, C->X, T->U; skip N")
    lines.append(f"  N bases skipped: {skipped_n}")
    lines.append(f"  RNA (transcribed) length: {len(rna_from_dna)}\n")

    lines.append("Translation (RNA -> Protein):")
    lines.append(f"  Codon length k: {k}")
    lines.append(f"  Frame selected (min conflicts): {best_frame}")
    lines.append(f"  Unambiguous codons available: {len(translation_table)}")
    lines.append(f"  Conflicting codons in inferred table: {conflicts}")
    lines.append(f"  Codons translated from transcribed RNA: {codons_used}")
    lines.append(f"  Unknown/ambiguous codons ('?') produced: {unknown_codons}\n")

    lines.append("Comparison (predicted vs given protein):")
    lines.append(f"  Predicted protein length: {stats['pred_len']}")
    lines.append(f"  Overlap compared: {stats['overlap']}")
    lines.append(f"  Comparable positions (pred != '?'): {stats['comparable']}")
    lines.append(f"  Matches: {stats['matches']}")
    lines.append(f"  Mismatches: {stats['mismatches']}")
    lines.append(f"  Unknowns in overlap: {stats['unknowns']}")
    lines.append(f"  Identity (matches/comparable): {stats['identity_percent']:.2f}%\n")

    #Snippets
    lines.append("Snippets (first 120 chars):")
    lines.append(f"  RNA_transcribed: {rna_from_dna[:120]}")
    lines.append(f"  Protein_pred:    {pred_protein[:120]}")
    lines.append(f"  Protein_given:   {protein_given[:120]}")

    out_path = out_dir / "task5_translation_check.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[Task 5] Translation check written to: {out_path}")




#---------------- MAIN ----------------

def main():
    dna = read_fasta(DATA_DIR / "gene_a.fa")
    rna = read_fasta(DATA_DIR / "rna_a.fa")
    protein = read_fasta(DATA_DIR / "protein_a.fa")

    run_task1(dna, rna, protein)
    run_task2(dna, rna)
    run_task3(rna, protein)
    run_task4(rna, protein, k=3)
    run_task5(dna, rna, protein, k=3)


if __name__ == "__main__":
    main()
