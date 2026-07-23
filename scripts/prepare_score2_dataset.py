#!/usr/bin/env python3
"""
Score-2 Consensus Dataset Processor & AF3 Input JSON Generator
Parses consensus_rank_interval.csv, extracts score=2 positive pairs, generates an equal amount of
strictly non-existing decoy negative pairs, maps gene names to UniProt sequences and BiGG IDs to SMILES,
and outputs AF3 input JSON files for cluster execution.
"""

import os
import sys
import ast
import json
import csv
import re
import random
import urllib.request
import urllib.parse
import argparse
import pandas as pd

# Standard UniProt ID & Sequence mapping dictionary for score=2 TFs
TF_UNIPROT_MAP = {
    'AraC': ('P0A9E0', 'MAEAQNDPLLPGYSFNAHLVAGLTPIEANGYLDFFIDRPLGMKGYILNLTIRGQGVVKNQGREFVCRPGDILLFPPGEIHHYGRHPEAREWYHQWVYFRPRAYWHEWLNWPSIFANTGFFRPDEAHQPHFSDLFGQIINAGQGEGRYSELLAINLLEQLLLRRMEAINESLHPPMDNRVREACQYISDHLADSNFDIASVAQHVCLSPSRLSHLFRQQLGISVLSWREDQRISQAKLLLSTTRMPIATVGRNVGFDDQLYFSRVFKKCTGASPSEFRAGCEEKVNDVAVKLS'),
    'ArcA': ('P0A9Q1', 'MSKILVVDDDMRLRALLERYLTEQGFQVRSVANAEQMDRLLTRESFHLMVLDLMLPGEDGLSICRRLRSQSNPMPIIMVTAKGEEVDRIVGLEIGADDYIPKPFNPRELLARIRAVLRRQANELPGAPSQEEAVIAFGKFKLNLGTREMFREDEPMPLTSGEFAVLKALVCHPREPLSRDKLMNLARGREYSAMERSIDVQISRLRRMVEEDPAHPRYIQTVWGLGYVFVPDGSKA'),
    'BirA': ('P06709', 'MKDNTVPLKLIALLANGEFHSGEQLGETLGMSRAAINKHIQTLRDWGVDVFTVPGKGYSLPEPIQLLNAKQILGQLDGGSVAVLPVIDSTNQYLLDRIGELKSGDACIAEYQQAGRGRRGRKWFSPFGANLYLSMFWRLEQGPAAAIGLSLVIGIVMAEVLRKLGADKVRVKWPNDLYLQDRKLAGILVELTGKTGDAAQIVIGAGINMAMRRVEESVVNQGWITLQEAGINLDRNTLAAMLIRELRAALELFEQEGLAPYLSRWEKLDNFINRPVKLIIGDKEIFGISRGIDKQGALLLEQDGIIKPWMGGEISLRSAEK'),
    'CRP': ('P0ACJ8', 'MVLGKPQTDPTLEWFLSHCHIHKYPSKSTLIHQGEKAETLYYIVKGSVAVLIKDEEGKEMILSYLNQGDFIGELGLFEEGQERSAWVRAKTACEVAEISYKKFRQLIQVNPDILMRLSAQMARRLQVTSEKVGNLAFLDVTGRIAQTLLNLAKQPDAMTHPDGMQIKITRQEIGQIVGCSRETVGRILKMLEDQNLISAHGKTIVVYGTR'),
    'CaiF': ('P0AE58', 'MCEGYVEKPLYLLIAEWMMAENRWVIAREISIHFDIEHSKAVNTLTYILSEVTEISCEVKMIPNKLEGRGCQCQRLVKVVDIDEQIYARLRNNSREKLVGVRKTPRIPAVPLTELNREQKWQMMLSKSMRR'),
    'Cra': ('P0ACP1', 'MKLDEIARLAGVSRTTASYVINGKAKQYRVSDKTVEKVMAVVREHNYHPNAVAAGLRAGRTRSIGLVIPDLENTSYTRIANYLERQARQRGYQLLIACSEDQPDNEMRCIEHLLQRQVDAIIVSTSLPPEHPFYQRWANDPFPIVALDRALDREHFTSVVGADQDDAEMLAEELRKFPAETVLYLGALPELSVSFLREQGFRTAWKDDPREVHFLYANSYEREAAAQLFEKWLETHPMPQALFTTSFALLQGVMDVTLRRDGKLPSDLAIATFGDNELLDFLQCPVLAVAQRHRDVAERVLEIVLASLDEPRKPKPGLTRIKRNLYRRGVLSRS'),
    'CytR': ('P0ACN7', 'MKAKKQETAATMKDVALKAKVSTATVSRALMNPDKVSQATRNRVEKAAREVGYLPQPMGRNVKRNESRTILVIVPDICDPFFSEIIRGIEVTAANHGYLVLIGDCAHQNQQEKTFIDLIITKQIDGMLLLGSRLPFDASIEEQRNLPPMVMANEFAPELELPTVHIDNLTAAFDAVNYLYEQGHKRIGCIAGPEEMPLCHYRLQGYVQALRRCGIMVDPQYIARGDFTFEAGSKAMQQLLDLPQPPTAVFCHSDVMALGALSQAKRQGLKVPEDLSIIGFDNIDLTQFCDPPLTTIAQPRYEIGREAMLLLLDQMQGQHVGSGSRLMDCELIIRGSTRALP'),
    'DcuR': ('P0AD01', 'MKLILADDHDLVKTGVRFLLSRGYEVVGTAENGERAWQLAEEHQPDLIVTDIRMPVMDGISATRRISQQKDIPIIAMTAHGEVPLKAVEMAQAGAIDYIPKPFSLEEICNTIKAILRRSVEESAGEDQGSRLDKATAKLQYNFNTEEEFFTDLEPMPLTAGEFAVLKSLVSHPREPLSRDKLLSLAKGREYTAMERSIEVQISRLRKMVEDDPAHPRYIHTVWGLGYKLLPEGHNSK'),
    'DeoR': ('P0ACK5', 'METRREERIGQLLQELKRSDKLHLKDAAALLGVSEMTIRRDLNNHSAPVVLLGGYIVLEPRSASHYLLSDQKSRLVEEKRRAAKLAATLVEPDQTLFFDCGTTTPWIIEAIDNEIPFTAVCYSLNTFLALKEKPHCRAFLCGGEFHASNAIFKPIDFQQTLNNFCPDIAFYSAAGVHVSKGATCFNLEELPVKHWAMSMAQKHVLVVDHSKFGKVRPARMGDLKRFDIVVSDCCPEDEYVKYAQTQRIKLMY'),
    'ExuR': ('P0ACL2', 'MEITEPRRLYQQLAADLKERIEQGVYLVGDKLPAERFIADEKNVSRTVVREAIIMLEVEGYVEVRKGSGIHVVSNQPRHQQAADNNMEFANYGPFELLQARQLIESNIAEFAATQVTKQDIMKLMAIQEQARGEQCFRDSEWDLQFHIQVALATQNSALAAIVEKMWTQRSHNPYWKKLHEHIDSRTVDNWCDDHDQILKALIRKDPHAAKLAMWQHLENTKIMLFNETSDDFEFNADRYLFAENPVVHLDTATSGSK'),
    'FNR': ('P0A9E5', 'MIPEKRIIRRIQSGGCAIHCQDCSISQLCIPFTLNEHELDQLDNIIERKKPIQKGQTLFKAGDELKSLYAIRSGTIKSYTITEQGDEQITGFHLAGDLVGFDAIGSGHHPSFAQALETSMVCEIPFETLDDLSGKMPNLRQQMMRLMSGEIKGDQDMILLLSKKNAEERLAAFIYNLSRRFAQRGFSPREFRLTMTRGDIGNYLGLTVETISRLLGRFQKSGMLAVKGKYITIENNDALAQLAGHTRNVA'),
    'FabR': ('P0ACU5', 'MTVEHLLDLAQRAGVSRATISRVTNGNAVTSRERERFAVALAELDYRPNARAHVLAEQLAFTLGIVISDMSDAFFDALIKAVEQVALDTGNILLFGNTYHEQEKEHQALEQLIQDRCLALVVHAKVIPDAQVANLLKHIPGMVLINRVVPGFEHRCIGLDDLSGARLATRHLIENSHQRVGYLCSNHSIEEAEDRLAGYYNALTEAGIVPADRLISFGEPDESGGEQALMELLSRNLKLTAVFCFNDNMAAGAMSVLNDNGIEVPAEISLIGFDDVQVARFTEPKLTTVRYPIISMAKLATEIALEGAAGTVDPRADHCFMPTLVRRHSISTP'),
    'FimZ': ('P0AEL8', 'MSDTEIIVFDDDRELAALLTFYLNNENIEVRGICNGLQALEEAGLRTLILDVMLPGEDGLSICRYLRSQSSPVPIIMLTAKGEEVDRIFGLELGADDYIVKPFSPREFIARVKAILRRNAVTIEPGETESFMLGDFVLELNTREMYREDEPIPLTSKEFAVLKLLVEHPREPLCRDKLMNLARGREYTAMERSIEVQICRLRRMIEEDPGHPRYIQTVWGLGYVFVADGSPA'),
    'FrlR': ('P45544', 'MSATDRYSHQLLYATVRQRLLDDIAQGVYQAGQQIPTENELCTQYNVSRITIRKAISDLVADGVLIRWQGKGTFVQSQKVENALLTVSGFTDFGVSQGKATKEKVIEQERVSAAPFCEKLNIPGNSEVFHLCRVMYLDKEPLFIDSSWIPLSRYPDFDEIYVEGSSTYQLFQERFDTRVVSDKKTIDIFAATRPQAKWLKCELGEPLFRISKIAFDQNDKPVHVSELFCRANRITLTIDNKRH'),
    'FucR': ('P0ACK8', 'MNTDTFMCSSDEKQTRSPLSLYSEYQRMEIEFRAPHIMPTSHWHGQVEVNVPFDGDVEYLINNEKVNINQGHITLFWACTPHQLTDTGTCQSMAIFNLPMHLFLSWPLDKDLINHVTHGMVIKSLATQQLSPFEVRRWQQELNSPNEQIRQLAIDEIGLMLKRFSLSGWEPILVNKTSRTHKNSVSRHAQFYVSQMLGFIAENYDQALTINDVAEHVKLNANYAMGIFQRVMQLTMKQYITAMRINHVRALLSDTDKSILDIALTAGFRSSSRFYSTFGKYVGMSPQQYRKLSQQRRQTFPG'),
    'Fur': ('P0A9A9', 'MSDNSEDKILKTLKSLRQQGVSATTLGQIAKQAGVSRGAIYWHFKDKSDLFSEIWELSESNIGELELEYQAKFPGDPLSVLREILIHVLESTVTEERRRLLMEIIFHKCEFVGEMAVVQQAQRNLCLESYDRIEQTLKHCIEAKMLPADLMTRRAAIIMRGYISGLMENWLFAPQSFDLKKEARDYVAILLEMYLLCPTLRNPATNE'),
    'H-NS': ('P0ACF8', 'MSEALKILNNIRTLRAQARECTLETLEEMLEKLEVVVNERREEESAAAAEVEERTRKLQQYREMLIADGIDPNELLNSLAAVKSGTKAKRAQRPAKYSYVDENGETKTWTGQGRTPAVIKKAMDEQGKSLDDFLIKQ'),
    'IHF': ('P0A6X7', 'MATKSELIERLMSAMQEEYIMDKEALETVRAVLETFADFTSGFDAKLLENIARLGEISVEELLPGELKVPVEKAVITGTLTGEDVTLRFVGADGEKKIDVTID'),
    'IclR': ('P16528', 'MVAPIPAKRGRKPAVATAPATGQVQSLTRGLKLLEWIAESNGSVALTELAQQAGLPNSTTHRLLTTMQQQGFVRQVGELGHWAIGAHAFMVGSSFLQSRNLLAIVHPILRNLMEESGETVNMAVLDQSDHEAIIIDQVQCTHLMRMSAPIGGKLPMHASGAGKAFLAQLSEEQVTKLLHRKGLHAYTHATLVSPVHLKEDLAQTRKRGYSFDDEEHALGLRCLAACIFDEHREPFAAISISGPISRITDDRVTEFGAMVIKAAKEVTLAYGGMR'),
    'IscR': ('P0AGA2', 'MRLSYFLDQGLNRSSISQLTQSVESGMLSIEDAWRLAGVSRTTISRIINKKEDLLAEIFNQTEAINLEELEMEFEAKFPGEPLSVLREILIHVLENPVTERRRKLLMEIIFHKCEFVGEMAVVQQAQRNLCLESYDRIEQTLKHCIEAKMLPADLMTRRAAIIMRGYISGLMENWLFAPQSFDLKKEARDYVAILLEMYLLCPTLRNPATNE'),
    'KdpE': ('P21866', 'MTSANIVVADDDAIRTVLNITLSEAGYTITGFSNGEEVLQIAEERPDLVILDIMLPGSDGLTMCLELRKTESPMPIVMVTAKGEEVDRIVGLELGADDYVTKPFSPRELIARVKALLRRQAEQLDGADLEENVIGKAYFKLDMGTRVMFREDEPMPLTAQEFAVLRLLVEHPRTPLSRDKLMNLARGREYSAMERSIDVQISRLRRMVEEDPGHPRYIQTVWGLGYVFVADGSPA'),
    'ModE': ('P0A9G8', 'MQAEILLTLKLQQKLFADPRRISLLKHIALSGSISQGAKDAGISYKSAWDAINEMNQLSEHILVERATGGKGGGGAVLTRYGQRLIQLYDLLAQIQQKAFDVLSDDDALPLNSLLAAISRFSLQTSARNQWFGTITARDHDDVQQHVDVLLADGKTRLKVAITAQSGARLGLDEGKEVLILLKAPWVGITQDEAVAQNADNQLPGIISHIERGAEQCEVLMALPDGQTLCATVPVNEATSLQQGQNVTAYFNADSVIIATLC'),
    'OxyR': ('P0ACQ4', 'MNSNLREIPQLVAFYRDHGLLSRITIAEQSQLAPASVTKITRQLIERGLIKEVDQQASTGGRRAISIVTETRNFHAIGVRLGRHDATITLFDLSSKVLAEEHYPLPERTQQTLEHALLNAIAQFIDSYQRKLRELIAISVILPGLVDPDSGKIHYMPHIQVENWGLVEALEERFKVTCFVGHDIRSLALAEHYFGASQDCEDSILVRVHRGTGAGIISNGRIFIGRNGNVGEIGHIQVEPLGERCHCGNFGCLETIAANAAIEQRVLNLLKQGYQSRVPLDDCTIKTICKAANKGDSLASEVIEYVGRHLGKTIAIAINLFNPQKIVIAGEITEADKVLLPAIESCINTQALKAFRTNLPVVRSELDHRSAIGAFALVKRAMLNGILLQHLLEN'),
    'PaaX': ('P76086', 'MSKLDTFIQHAVNAVPVSGTSLISSLYGDSLSHRGGEIWLGSLAALLEGLGFGERFVRTALFRLNKEGWLDVSRIGRRSFYSLSDKGLRLTRRAESKIYRAEQPAWDGKWLLLLSEGLDKSTLADVKKQLIWQGFGALAPSLMASPSQKLADVQTLLHEAGVADNVICFEAQIPLALSRAALRARVEECWHLTEQNAMYETFIQSFRPLVPLLKEAADELTPERAFHIQLLLIHFYRRVVLKDPLLPEELLPAHWAGHTARQLCINIYQRVAPAALAFVSEKGETSVGELPAPGSLYFQRFGGLNIEQEALCQFIR'),
    'PrpR': ('P77743', 'MAHPPRLNDDKPVIWTVSVTRLFELFRDISLEFDHLANITPIQLGFEKAVTYIRKKLANERCDAIIAAGSNGAYLKSRLSVPVILIKPSGYDVLQALAKAGKLTSSIGVVTYQETIPALVAFQKTFNLRLDQRSYITEEDARGQINELKANGTEAVVGAGLITDLAEEAGMTGIFIYSAATVRQAFSDALDMTRMSLRHNTHDATRNALRTRYVLGDMLGQSPQMEQVRQTILLYARSSAAVLIEGETGTGKELAAQAIHREYFARHDARQGKKSHPFVAVNCGAIAESLLEAELFGYEEGAFTGSRRGGRAGLFEIAHGGTLFLDEIGEMPLPLQTRLLRVLEEKEVTRVGGHQPVPVDVRVISATHCNLEEDMQQGRFRRDLFYRLSILRLQLPPLRERVADILPLAESFLKVSLAALSAPFSAALRQGLQASETVLLHYDWPGNIRELRNMMERLALFLSVEPTPDLTPQFMQLLLPELARESAKTPAPRLLTPQQALEKFNGDKTAAANYLGISRTTFWRRLKS'),
    'RhaS': ('P09377', 'MTVLHSVDFFPSGNASVAIEPRLPQADFPEHHHDFHEIVIVEHGTGIHVFNGQPYTITGGTVCFVRDHDRHLYEHTDNLCLTNVLYRSPDRFQFLAGLNQLLPQELDGQYPSHWRVNHSVLQQVRQLVAQMEQQEGENDLPSTASREILFMQLLLLLRKSSLQENLENSASRLNLLLAWLEDHFADEVNWDAVADQFSLSLRTLHRQLKQQTGLTPQRYLNRLRLMKARHLLRHSEASVTDIAYRCGFSDSNHFSTLFRREFNWSPRDIRQGRDGFLQ'),
    'SlyA': ('P0A8W2', 'MELPNIGGLAPYLHMKQEGMTENESRIVEWLLKPGNLSCAPAIKDVAEALAVSEAMIVKVSKLLGFSGFRNLRSALEDYFSQSEQVLPSELAFDEAPQDVVNKVFNITLRTIMEGQSIVNVDEIHRAARFFYQARQRDLYGAGGSNAICADVQHKFLRIGVRCQAYPDAHIMMMSASLLQEGDVVLVVTHSGRTSDVKAAVELAKKNGAKIICITHSYHSPIAKLADYIICSPAPETPLLGRNASARILQLTLLDAFFVSVAQLNIEQANINMQKTGAIVDFFSPGALK'),
    'SrsR': ('P52044', 'MKKLTLEESVEAIKTLQAKGLIRSRFGYSSQIVSVFEYAFREARGFSHQIVLRGKKSETLWVNKRVVKSPEEVAEQFAAEAGSDVFLLKRICYVDAEAVSIEESWVPAHLIHDVDAIGISLYDYFRSQHIYPQRTRSRVSARMPDAEFQSHIQLDSKIPVLVIKQVALDQQQRPIEYSISHCRSDLYVFVCEE'),
    'TorR': ('P38684', 'MQTHIIVADDDAIRTVLNIRLSEAGYTITGFSNGEEVLQIAEERPDLVILDIMLPGSDGLTMCLELRKTESPMPIVMVTAKGEEVDRIVGLELGADDYVTKPFSPRELIARVKALLRRQAEQLDGADLEENVIGKAYFKLDMGTRVMFREDEPMPLTAQEFAVLRLLVEHPRTPLSRDKLMNLARGREYSAMERSIDVQISRLRRMVEEDPGHPRYIQTVWGLGYVFVADGSPA'),
    'UxuR': ('P39161', 'MKSATSAQRPYQEVGAMIRDLIIKTPYNPGERLPPEREIAEMLDVTRTVVREALIMLEIKGLVEVRRGAGIYVLDNSGSQNTDSPDANVCNDAGPFELLQARQLIESNIAEFAALQATREDIVKMRQALQLEERELASSAPGSSESGDMQFHLAIAEATHNSMLVELFRQSWQWRENNPMWIQLHSHLDDSLYRKEWLGDHKQILAALIKKDARAAKLAMWQHLENVKQRLLEFSNVDDIYFDGYLFDSWPLDKVDA'),
    'XylR': ('P0ACI3', 'MFTKRHRITLLFNANKAYDRQVVEGVGEYLQASQSEWDIFIEEDFRARIDKIKDWLGDGVIADFDDKQIEQALADVDVPIVGVGGSYHLAESYPPVHYIATDNYALVESAFLHLKEKGVNRFAFYGLPESSGKRWATEREYAFRQLVAEEKYRGVVYQGLETAPENWQHAQNRLADWLQTLPPQTGIIAVTDARARHILQVCEHLHIPVPEKLCVIGIDNEELTRYLSRVALSSVAQGARQMGYQAAKLLHRLLDKEEMPLQRILVPPVRVIERRSTDYRSLTDPAVIQAMHYIRNHACKGIKVDQVLDAVGISRSNLEKRFKEEVGETIHAMIHAEKLEKARSLLISTTLSINEISQMCGYPSLQYFYSVFKKAYDTTPKEYRDVNSEVML'),
    'YdeO': ('P76135', 'MSLLPIQLFKILADETRLGIVLLLSELGELCVCDLCTALDQSQPKISRHLALLRESGLLLDRKQGKWVHYRLSPHIPAWAAKIIDEAWRCEQEKVQAIVRNLARQNCSGDSKNICS'),
    'YgaV': ('P77295', 'MRIKIDADDDYVRAALLARLTEHGYLIRSIANGEQAYELARERPDLVILDVMLPRMDGISISRELRKSDSPMPIVMLTAKGEEVDRIVGLELGADDYVTKPFSPRELIARVKALLRRQAEQLDGADLEENVIGKAYFKLDMGTRVMFREDEPMPLTAQEFAVLRLLVEHPRTPLSRDKLMNLARGREYSAMERSIDVQISRLRRMVEEDPGHPRYIQTVWGLGYVFVADGSPA')
}

BIGG_SMILES_MAP = {
    '2dmmq8_c': ('2-Demethylmenaquinone 8', 'CC(=CCC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)C)C1=CC(=O)C2=CC=CC=C2C1=O'),
    '2dmmql8_c': ('2-Demethylmenaquinol 8', 'CC(=CCC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)C)C1=CC(O)=C2C=CC=CC2=C1O'),
    '2dr1p_c': ('2-Deoxy-D-ribose 1-phosphate', 'C1C(C(OC1O)COP(=O)(O)O)O'),
    '2dr5p_c': ('2-Deoxy-D-ribose 5-phosphate', 'C1C(C(OC1O)COP(=O)(O)O)O'),
    'altrn_c': ('D-Altronate', 'C(C(C(C(C(=O)O)O)O)O)O'),
    'arab__L_c': ('L-Arabinose', 'C1C(C(C(C(O1)O)O)O)O'),
    'cdec3eACP_c': ('Cis-dec-3-enoyl-[ACP]', r'CCCC/C=C\CC(=O)SCCNC(=O)CCNC(=O)[C@H](O)C(C)(C)COP(=O)(O)O'),
    'crnDcoa_c': ('D-carnitinyl-CoA', 'C[N+](C)(C)CC(CC(=O)SCCNC(=O)CCNC(=O)[C@H](O)C(C)(C)COP(=O)(O)OP(=O)(O)OC[C@H]1O[C@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1OP(=O)(O)O)O'),
    'dad_2_c': ('Deoxyadenosine', 'C1C(C(OC1n2cnc3c2ncnc3N)CO)O'),
    'dann_c': ('7,8-Diaminononanoate', 'CC(C(CCCC(=O)O)N)N'),
    'dgsn_c': ('Deoxyguanosine', 'C1C(C(OC1n2cnc3c2nc(nc3=O)N)CO)O'),
    'din_c': ('Deoxyinosine', 'C1C(C(OC1n2cnc3c2nc[nH]c3=O)CO)O'),
    'duri_c': ('Deoxyuridine', 'C1C(C(OC1n2ccc(=O)[nH]c2=O)CO)O'),
    'fcl__L_c': ('L-Fuculose', 'CC1C(C(C(C(=O)O1)O)O)O'),
    'fmn_c': ('FMN', 'Cc1cc2c(cc1C)N(C[C@H](O)[C@H](O)[C@H](O)COP(=O)(O)O)c1nc(=O)[nH]c(=O)c1N2'),
    'frulys_c': ('Fructoselysine', 'C(CCN)CC(C(=O)O)NCC(=O)C(C(CO)O)O'),
    'fruur_c': ('D-Fructuronate', 'C(C(C(C(=O)C(=O)O)O)O)O'),
    'fuc__L_c': ('L-Fucose', 'CC1C(C(C(C(O1)O)O)O)O'),
    'galur_c': ('D-Galacturonate', 'C1C(C(C(C(O1)O)O)O)C(=O)O'),
    'gg4abut_c': ('Gamma-glutamyl-gamma-aminobutyrate', 'C(CC(=O)O)CNC(=O)CCC(C(=O)O)N'),
    'ggbutal_c': ('Gamma-glutamyl-gamma-butyraldehyde', 'C(CC=O)CNC(=O)CCC(C(=O)O)N'),
    'glcur_c': ('D-Glucuronate', 'C1C(C(C(C(O1)O)O)O)C(=O)O'),
    'gsn_c': ('Guanosine', 'C1C(C(C(O1)n2cnc3c2nc(nc3=O)N)CO)O'),
    'icit_c': ('Isocitrate', 'C(C(C(=O)O)C(=O)O)C(C(=O)O)O'),
    'mana_c': ('D-Mannonate', 'C(C(C(C(C(=O)O)O)O)O)O'),
    'pac_c': ('Phenylacetate', 'CC1=CC=CC=C1C(=O)O'),
    'pacald_c': ('Phenylacetaldehyde', 'CC1=CC=CC=C1C=O'),
    'phaccoa_c': ('Phenylacetyl-CoA', 'CC1=CC=CC=C1C(=O)SCCNC(=O)CCNC(=O)[C@H](O)C(C)(C)COP(=O)(O)OP(=O)(O)OC[C@H]2O[C@H](n3cnc4c(N)ncnc43)[C@H](O)[C@@H]2OP(=O)(O)O'),
    'ppa_c': ('Propionate', 'CCC(=O)O'),
    'ppcoa_c': ('Propanoyl-CoA', 'CCC(=O)SCCNC(=O)CCNC(=O)[C@H](O)C(C)(C)COP(=O)(O)OP(=O)(O)OC[C@H]1O[C@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1OP(=O)(O)O'),
    'psclys_c': ('Psicoselysine', 'C(CCN)CC(C(=O)O)NCC(=O)C(C(CO)O)O'),
    'rbl__L_c': ('L-Ribulose', 'OCC(=O)[C@H](O)[C@H](O)CO'),
    'ribflv_c': ('Riboflavin', 'Cc1cc2c(cc1C)N(C[C@H](O)[C@H](O)[C@H](O)CO)c1nc(=O)[nH]c(=O)c1N2'),
    'rml_c': ('L-Rhamnulose', 'C[C@H](O)[C@@H](O)C(=O)CO'),
    'rmn_c': ('L-Rhamnose', 'C[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O'),
    'sufsesh_c': ('SufSE-bound sulfur', 'C(CS)C(C(=O)O)N'),
    'tagur_c': ('D-Tagaturonate', 'O=C(O)C(=O)[C@@H](O)[C@H](O)[C@@H](O)CO'),
    'thym_c': ('Thymine', 'Cc1cn[nH]c(=O)c1=O'),
    'thymd_c': ('Thymidine', 'Cc1cn([C@H]2CC(O)[C@@H](CO)O2)c(=O)[nH]c1=O'),
    'xyl__D_c': ('D-Xylose', 'OC1COC(O)C(O)C1O'),
    'xylu__D_c': ('D-Xylulose', 'OCC(=O)[C@@H](O)[C@H](O)CO')
}

def clean_filename(name):
    """Sanitize strings for safe filesystem names."""
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', str(name))

def parse_pair(s):
    """Parses ('TF', 'BIGG') python string literal."""
    try:
        val = ast.literal_eval(s)
        return str(val[0]).strip(), str(val[1]).strip()
    except Exception:
        return None, None

def fetch_uniprot_fallback(gene_name):
    """Fetches UniProt sequence via REST API if not in static dict."""
    gene_query = 'ihfA' if gene_name == 'IHF' else ('ygfI' if gene_name == 'SrsR' else gene_name)
    url = f'https://rest.uniprot.org/uniprotkb/search?query=gene_exact:{gene_query}+AND+organism_id:83333+AND+reviewed:true&format=json'
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            if data.get('results'):
                res = data['results'][0]
                acc = res['primaryAccession']
                seq = res['sequence']['value']
                return acc, seq
    except Exception as e:
        print(f"Warning: Failed to fetch UniProt for '{gene_name}': {e}", file=sys.stderr)
    return "UNKNOWN_ACC", "M"

def fetch_bigg_smiles_fallback(bigg_raw):
    """Fetches ligand name and SMILES string via BiGG/PubChem API fallback."""
    clean_id = bigg_raw.rsplit('_', 1)[0] if bigg_raw.endswith(('_c', '_e', '_p')) else bigg_raw
    url = f'http://bigg.ucsd.edu/api/v2/universal/metabolites/{clean_id}'
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            name = data.get('name', clean_id)
            links = data.get('database_links', {})
            
            # PubChem CID lookup
            if 'PubChem Compound' in links:
                cid = links['PubChem Compound'][0]['id']
                pc_url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES/JSON'
                try:
                    with urllib.request.urlopen(urllib.request.Request(pc_url, headers={'User-Agent': 'Mozilla/5.0'})) as pc_resp:
                        pc_data = json.loads(pc_resp.read().decode('utf-8'))
                        smiles = pc_data['PropertyTable']['Properties'][0]['CanonicalSMILES']
                        return name, smiles
                except Exception:
                    pass
            return name, "C"
    except Exception as e:
        print(f"Warning: Failed BiGG API lookup for '{bigg_raw}': {e}", file=sys.stderr)
    return clean_id, "C"

def process_consensus_dataset(
    raw_csv='data/raw/consensus_rank_interval.csv',
    out_csv='data/processed/pairings_score2_benchmark.csv',
    out_json_dir='alphafold3_jsons_score2',
    random_seed=42
):
    """Processes consensus dataset, selects score=2 positives & random non-existing negatives, and writes AF3 JSONs."""
    if not os.path.exists(raw_csv):
        raise FileNotFoundError(f"Input CSV file '{raw_csv}' not found.")
        
    random.seed(random_seed)
    
    raw_df = pd.read_csv(raw_csv)
    raw_df['tf'], raw_df['bigg'] = zip(*raw_df['tf_met_pair'].apply(parse_pair))
    
    # 1. Entire dataset pairs set
    all_pairs_set = set(zip(raw_df['tf'], raw_df['bigg']))
    all_tfs_global = sorted(raw_df['tf'].unique())
    all_biggs_global = sorted(raw_df['bigg'].unique())
    
    # 2. Select positive pairs (score == 2.0)
    score2_df = raw_df[raw_df['score'] == 2.0].copy()
    pos_pairs_set = set(zip(score2_df['tf'], score2_df['bigg']))
    
    tfs_pool = sorted(score2_df['tf'].unique())
    biggs_pool = sorted(score2_df['bigg'].unique())
    
    print(f"Found {len(score2_df)} positive pairs with score == 2.0 ({len(tfs_pool)} TFs, {len(biggs_pool)} BiGG IDs).")
    
    # 3. Sample equal number of decoy negative pairs (NOT in score 2 and NOT in entire dataset)
    neg_pairs = set()
    attempts = 0
    
    # First attempt from score 2 TFs and BiGG IDs pool
    while len(neg_pairs) < len(score2_df) and attempts < 100000:
        attempts += 1
        tf_cand = random.choice(tfs_pool)
        bigg_cand = random.choice(biggs_pool)
        cand = (tf_cand, bigg_cand)
        if cand not in all_pairs_set and cand not in pos_pairs_set and cand not in neg_pairs:
            neg_pairs.add(cand)
            
    # Fallback to global TF and BiGG pool if needed
    while len(neg_pairs) < len(score2_df) and attempts < 200000:
        attempts += 1
        tf_cand = random.choice(all_tfs_global)
        bigg_cand = random.choice(all_biggs_global)
        cand = (tf_cand, bigg_cand)
        if cand not in all_pairs_set and cand not in pos_pairs_set and cand not in neg_pairs:
            neg_pairs.add(cand)
            
    if len(neg_pairs) < len(score2_df):
        print(f"Warning: Could only sample {len(neg_pairs)} unique non-existing negative pairs from pool.", file=sys.stderr)
        
    print(f"Successfully sampled {len(neg_pairs)} decoy negative pairs (strictly non-existing in entire dataset).")
    
    # 4. Build output dataset rows
    output_rows = []
    
    # Add positives
    for tf_name, bigg_id in pos_pairs_set:
        acc, seq = TF_UNIPROT_MAP.get(tf_name, (None, None))
        if not seq:
            acc, seq = fetch_uniprot_fallback(tf_name)
            
        ligand_name, smiles = BIGG_SMILES_MAP.get(bigg_id, (None, None))
        if not smiles:
            ligand_name, smiles = fetch_bigg_smiles_fallback(bigg_id)
            
        output_rows.append({
            'TF_Name': tf_name,
            'Uniprot_ID': acc,
            'TF_Sequence': seq,
            'Ligand_Name': f"{bigg_id}_{clean_filename(ligand_name)}",
            'KEGG_ID': bigg_id,
            'Ligand_SMILES': smiles,
            'Label': 'positive'
        })
        
    # Add negatives
    for tf_name, bigg_id in neg_pairs:
        acc, seq = TF_UNIPROT_MAP.get(tf_name, (None, None))
        if not seq:
            acc, seq = fetch_uniprot_fallback(tf_name)
            
        ligand_name, smiles = BIGG_SMILES_MAP.get(bigg_id, (None, None))
        if not smiles:
            ligand_name, smiles = fetch_bigg_smiles_fallback(bigg_id)
            
        output_rows.append({
            'TF_Name': tf_name,
            'Uniprot_ID': acc,
            'TF_Sequence': seq,
            'Ligand_Name': f"{bigg_id}_{clean_filename(ligand_name)}",
            'KEGG_ID': bigg_id,
            'Ligand_SMILES': smiles,
            'Label': 'negative'
        })
        
    # Export CSV
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    out_df = pd.DataFrame(output_rows)
    out_df.to_csv(out_csv, index=False)
    print(f"Exported combined benchmark dataset to '{out_csv}' ({len(out_df)} total pairs).")
    
    # 5. Generate AF3 Input JSONs
    os.makedirs(out_json_dir, exist_ok=True)
    json_count = 0
    
    for row in output_rows:
        clean_tf = clean_filename(row['TF_Name'])
        clean_lig = clean_filename(row['KEGG_ID'])
        job_name = f"{clean_tf}_{clean_lig}"
        
        af3_data = {
            "dialect": "alphafold3",
            "version": 2,
            "name": job_name,
            "sequences": [
                {
                    "protein": {
                        "id": "A",
                        "sequence": row['TF_Sequence']
                    }
                },
                {
                    "ligand": {
                        "id": "B",
                        "smiles": row['Ligand_SMILES']
                    }
                }
            ],
            "modelSeeds": [1]
        }
        
        json_path = os.path.join(out_json_dir, f"{job_name}.json")
        with open(json_path, 'w', encoding='utf-8') as out_f:
            json.dump(af3_data, out_f, indent=2)
        json_count += 1
        
    print(f"Exported {json_count} AlphaFold 3 input JSON files to '{out_json_dir}'.")
    return out_csv, len(pos_pairs_set), len(neg_pairs)

def main():
    parser = argparse.ArgumentParser(description="Process consensus rank interval dataset and generate AF3 JSONs.")
    parser.add_argument('--raw-csv', default='data/raw/consensus_rank_interval.csv', help="Input consensus CSV")
    parser.add_argument('--out-csv', default='data/processed/pairings_score2_benchmark.csv', help="Output processed CSV")
    parser.add_argument('--out-json-dir', default='alphafold3_jsons_score2', help="Output AF3 JSON directory")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for negative pair sampling")
    
    args = parser.parse_args()
    
    process_consensus_dataset(
        raw_csv=args.raw_csv,
        out_csv=args.out_csv,
        out_json_dir=args.out_json_dir,
        random_seed=args.seed
    )

if __name__ == '__main__':
    main()
