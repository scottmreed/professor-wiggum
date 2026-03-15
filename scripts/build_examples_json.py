#!/usr/bin/env python3
"""Build reaction_type_templates_examples.json from the rewrite JSON.

For each template, replaces R-group notation with specific simple molecules.
Adds both _example and _generic aliased fields so the existing render script works.
"""
import json
from datetime import datetime, timezone

EXAMPLES = {
    "rt_001": {
        "example_notes": "CH3CH2CH2-Cl (n-propyl chloride) undergoes Finkelstein exchange with NaI in acetone.",
        "current_state_example": ["CH3CH2CH2-Cl", "I-"],
        "resulting_state_example": ["CH3CH2CH2-I", "Cl-"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "CH3CH2CH2-Cl.I->CH3CH2CH2-I.Cl-",
                "note": "Iodide attacks n-propyl carbon; chloride departs (SN2). Classic Finkelstein in acetone."
            }
        ]
    },
    "rt_002": {
        "example_notes": "Ph-CH2CH2-OTs (2-phenylethyl tosylate) with KOH undergoes E2 to give styrene.",
        "current_state_example": ["Ph-CH2-CH2-OTs", "KOH"],
        "resulting_state_example": ["Ph-CH=CH2 (styrene)", "KOTs", "H2O"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "Ph-CH2-CH2-OTs.KOH>>Ph-CH=CH2.KOTs.H2O",
                "note": "KOH removes beta-H; pi bond forms; tosylate departs (E2). Product is styrene."
            }
        ]
    },
    "rt_003": {
        "example_notes": "Ph-CH2-CH2-CH2-Br (3-phenylpropyl bromide) with KOtBu gives Ph-CH2-CH=CH2 (allylbenzene).",
        "current_state_example": ["Ph-CH2-CH2-CH2-Br", "KOtBu"],
        "resulting_state_example": ["Ph-CH2-CH=CH2 (allylbenzene)", "Br-", "tBuOH"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "Ph-CH2-CH2-CH2-Br.KOtBu>>Ph-CH2-CH=CH2.Br-.tBuOH",
                "note": "KOtBu abstracts beta-H; alkene forms; bromide leaves (E2). Product is allylbenzene."
            }
        ]
    },
    "rt_004": {
        "example_notes": "Ethyl acetate (CH3CH2-O-C(=O)CH3) hydrolysis by NaOH gives ethanol and acetate.",
        "current_state_example": ["CH3CH2-O-C(=O)CH3 (ethyl acetate)", "NaOH"],
        "resulting_state_example": ["CH3CH2-OH (ethanol)", "CH3CO2Na (sodium acetate)"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "CH3CH2-O-C(=O)CH3.OH>>CH3CH2-OH.CH3CO2-",
                "note": "OH attacks carbonyl; tetrahedral intermediate collapses; ethoxide departs. Saponification."
            }
        ]
    },
    "rt_005": {
        "example_notes": "MeMgBr opens propylene oxide (propylene oxide, CH3-epoxide) at less substituted C to give 1-butanol.",
        "current_state_example": ["CH3MgBr", "propylene oxide (CH3CH-O-CH2)"],
        "resulting_state_example": ["CH3CH(O-)CH2CH3 → 1-butanol after workup"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "CH3MgBr.propylene oxide>>CH3CH(OMgBr)CH2CH3 -> workup -> 1-butanol",
                "note": "Methyl carbanion attacks less hindered epoxide carbon; ring opens; workup protonates alkoxide."
            }
        ]
    },
    "rt_006": {
        "example_notes": "Acetylacetone enolate (acetylacetonate) alkylated by ethyl bromoacetate.",
        "current_state_example": ["acetylacetonate anion (CH3COCH-COCH3)", "Br-CH2-CO2C2H5 (ethyl bromoacetate)"],
        "resulting_state_example": ["CH3CO-CH(CH2CO2Et)-COCH3 (alkylated acetylacetone ester)", "Br-"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "acetylacetonate.Br-CH2-CO2Et>>CH3CO-CH(CH2CO2Et)-COCH3.Br-",
                "note": "Enolate C attacks electrophilic CH2 of ethyl bromoacetate; bromide departs (SN2)."
            }
        ]
    },
    "rt_007": {
        "example_notes": "Allyl glucosinolate-derived thiohydroximate-O-sulfonate loses sulfate to give allyl isothiocyanate (mustard oil).",
        "current_state_example": ["CH2=CH-CH2-N=C(S-)(O-SO3-)"],
        "resulting_state_example": ["CH2=CH-CH2-N=C=S (allyl isothiocyanate)", "sulfate byproduct"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "CH2=CH-CH2-S-C(=N-OSO3-)>>CH2=CHCH2-NCS.SO4(2-) byproducts",
                "note": "N-O bond breaks; electron reorganization gives isothiocyanate cumulene. Mimics glucosinolate myrosinase product."
            }
        ]
    },
    "rt_008": {
        "example_notes": "PhS- (thiophenolate) displaces Cl from benzyl chloride to give benzyl phenyl sulfide.",
        "current_state_example": ["PhS- (sodium thiophenolate)", "Ph-CH2-Cl (benzyl chloride)"],
        "resulting_state_example": ["Ph-S-CH2-Ph (benzyl phenyl sulfide)", "Cl-"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "PhS-.Ph-CH2-Cl>>Ph-S-CH2-Ph.Cl-",
                "note": "Thiolate sulfur attacks benzyl carbon; chloride departs (SN2)."
            }
        ]
    },
    "rt_009": {
        "example_notes": "2-Hydroxypyridine anion (2-pyridyloxide) alkylated by allyl bromide to give 2-(allyloxy)pyridine.",
        "current_state_example": ["2-pyridyloxide (2-PyO-)", "CH2=CH-CH2-Br (allyl bromide)"],
        "resulting_state_example": ["2-(allyloxy)pyridine (2-PyO-CH2-CH=CH2)", "Br-"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "2-PyO-.CH2=CHCH2Br>>2-PyO-CH2CH=CH2.Br-",
                "note": "Pyridyloxide attacks allylic carbon; bromide leaves (SN2). Williamson allylation."
            }
        ]
    },
    "rt_010": {
        "example_notes": "Propargyl anion (HC≡C-) attacks allyl tosylate (CH2=CHCH2-OTs) to give pent-4-en-1-yne.",
        "current_state_example": ["HC#C- (acetylide anion)", "CH2=CH-CH2-OTs (allyl tosylate)"],
        "resulting_state_example": ["HC#C-CH2-CH=CH2 (pent-1-en-4-yne)", "-OTs"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "HC#C-.CH2=CHCH2-OTs>>HC#C-CH2CH=CH2.-OTs",
                "note": "Acetylide attacks allylic carbon; tosylate departs (SN2)."
            }
        ]
    },
    "rt_011": {
        "example_notes": "Trimethyl(2-methylpropyl)ammonium hydroxide (neopentyl-type) undergoes Hofmann to give isobutylene.",
        "current_state_example": ["CH3CH2-CH2-N+(CH3)3 OH- (n-propyltrimethylammonium hydroxide)"],
        "resulting_state_example": ["CH3CH=CH2 (propene)", "N(CH3)3 (trimethylamine)", "H2O"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "CH3CH2CH2-N+(CH3)3.OH>>CH3CH=CH2.N(CH3)3.H2O",
                "note": "OH removes beta-H; pi bond forms; trimethylamine leaves (Hofmann elimination gives less substituted alkene)."
            }
        ]
    },
    "rt_012": {
        "example_notes": "Acetic acid (CH3CO2H) methylation with CH2N2 gives methyl acetate (CH3CO2CH3).",
        "current_state_example": ["CH3CO2H (acetic acid)", "CH2N2 (diazomethane)"],
        "resulting_state_example": ["CH3CO2CH3 (methyl acetate)", "N2"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "CH3CO2H.CH2N2>>CH3CO2-.CH3-N2+",
                "note": "Proton transfer from acid to diazomethane generates ion pair intermediate."
            },
            {
                "step_index": 2,
                "reaction_example": "CH3CO2-.CH3-N2+>>CH3CO2CH3.N2",
                "note": "Carboxylate attacks methyl; N2 is expelled. Product is methyl acetate."
            }
        ]
    },
    "rt_013": {
        "example_notes": "Phenol (PhOH) O-methylated by diazomethane to give anisole (PhOCH3).",
        "current_state_example": ["PhOH (phenol)", "CH2N2 (diazomethane)"],
        "resulting_state_example": ["PhOCH3 (anisole)", "N2"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "PhOH.CH2N2>>PhO-.CH3-N2+",
                "note": "Phenol protonates diazomethane; phenoxide and methyldiazonium ion pair form."
            },
            {
                "step_index": 2,
                "reaction_example": "PhO-.CH3-N2+>>PhOCH3.N2",
                "note": "Phenoxide attacks methyl; N2 expelled. Product is anisole."
            }
        ]
    },
    "rt_014": {
        "example_notes": "Phenol + K2CO3 + ethyl iodide gives phenetole (PhOC2H5) via Williamson ether synthesis.",
        "current_state_example": ["PhOH (phenol)", "C2H5I (ethyl iodide)", "K2CO3"],
        "resulting_state_example": ["PhOC2H5 (phenetole)", "I-", "KHCO3"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "PhOH.K2CO3>>PhO-.KHCO3",
                "note": "K2CO3 deprotonates phenol to phenoxide."
            },
            {
                "step_index": 2,
                "reaction_example": "PhO-.C2H5I>>PhOC2H5.I-",
                "note": "Phenoxide attacks ethyl iodide (SN2); iodide departs. Product is phenetole."
            }
        ]
    },
    "rt_015": {
        "example_notes": "4-chlorobutan-1-ol with NaH undergoes intramolecular Williamson to give tetrahydrofuran (THF).",
        "current_state_example": ["HO-(CH2)4-Cl (4-chlorobutan-1-ol)", "NaH"],
        "resulting_state_example": ["tetrahydrofuran (THF)", "Cl-", "H2 (from NaH)"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "HO-(CH2)4-Cl.NaH>>NaO-(CH2)4-Cl.H2",
                "note": "NaH deprotonates alcohol to alkoxide; H2 released."
            },
            {
                "step_index": 2,
                "reaction_example": "NaO-(CH2)4-Cl>>THF.Cl-",
                "note": "Alkoxide attacks terminal C-Cl intramolecularly (SN2); five-membered ring closes."
            }
        ]
    },
    "rt_016": {
        "example_notes": "Ph-C(Cl)(CH3)2 (cumyl chloride) solvolysis in water gives Ph-C(OH)(CH3)2 (cumyl alcohol) via SN1.",
        "current_state_example": ["Ph-C(Cl)(CH3)2 (cumyl chloride)", "H2O"],
        "resulting_state_example": ["Ph-C(OH)(CH3)2 (cumyl alcohol)", "HCl"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "Ph-C(Cl)(CH3)2>>Ph-C+(CH3)2.Cl-",
                "note": "Ionization gives resonance-stabilized benzylic/tertiary carbocation."
            },
            {
                "step_index": 2,
                "reaction_example": "Ph-C+(CH3)2.H2O>>Ph-C(OH2+)(CH3)2",
                "note": "Water traps the carbocation to give the oxonium intermediate."
            },
            {
                "step_index": 3,
                "reaction_example": "Ph-C(OH2+)(CH3)2>>Ph-C(OH)(CH3)2.H+",
                "note": "Deprotonation gives neutral cumyl alcohol."
            }
        ]
    },
    "rt_017": {
        "example_notes": "Phenol + epichlorohydrin + NaOH gives phenyl glycidyl ether (2,3-epoxypropyl phenyl ether).",
        "current_state_example": ["PhOH (phenol)", "epichlorohydrin (ClCH2-CH-O-CH2)", "NaOH"],
        "resulting_state_example": ["PhO-CH2-CH(-O-)CH2 (phenyl glycidyl ether)", "Cl-", "H2O"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "PhOH.NaOH>>PhO-.H2O",
                "note": "NaOH generates phenoxide."
            },
            {
                "step_index": 2,
                "reaction_example": "PhO-.ClCH2CHOCH2>>PhO-CH2-CH(OH)-CH2-Cl",
                "note": "Phenoxide opens epoxide at less hindered carbon; chlorohydrin intermediate."
            },
            {
                "step_index": 3,
                "reaction_example": "PhO-CH2-CH(OH)-CH2-Cl.NaOH>>PhO-CH2-CH(-O-)CH2.Cl-.H2O",
                "note": "Base deprotonates alcohol; alkoxide closes ring intramolecularly (SN2); chloride departs."
            }
        ]
    },
    "rt_018": {
        "example_notes": "Methanol + acryloyl chloride + Et3N gives methyl acrylate (CH2=CH-CO2CH3).",
        "current_state_example": ["CH3OH (methanol)", "CH2=CH-COCl (acryloyl chloride)", "Et3N"],
        "resulting_state_example": ["CH2=CH-CO2CH3 (methyl acrylate)", "Et3N·HCl"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "CH3OH.CH2=CH-COCl>>tetrahedral intermediate",
                "note": "Methanol attacks acryloyl carbonyl; pi bond collapses to oxygen."
            },
            {
                "step_index": 2,
                "reaction_example": "tetrahedral intermediate>>CH2=CH-CO2CH3.Cl-.H+",
                "note": "Carbonyl reforms; chloride departs. Methyl acrylate forms."
            },
            {
                "step_index": 3,
                "reaction_example": "CH2=CH-CO2CH3.H+.Et3N>>CH2=CH-CO2CH3.Et3N-H+",
                "note": "Triethylamine scavenges HCl."
            }
        ]
    },
    "rt_019": {
        "example_notes": "Chlorobenzene + NaNH2 (sodamide) gives aniline via benzyne intermediate.",
        "current_state_example": ["PhCl (chlorobenzene)", "NaNH2", "strong base (KNH2/liq. NH3)"],
        "resulting_state_example": ["PhNH2 (aniline)", "Cl-"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "PhCl.NaNH2>>ortho-carbanion_PhCl",
                "note": "Sodamide removes ortho-H adjacent to Cl to give aryl carbanion."
            },
            {
                "step_index": 2,
                "reaction_example": "ortho-carbanion_PhCl>>benzyne.Cl-",
                "note": "Carbanion forms strained benzyne triple-bond surrogate; Cl- expelled."
            },
            {
                "step_index": 3,
                "reaction_example": "benzyne.NH2->>Ph(NH2)carbanion",
                "note": "Amide adds to benzyne; sigma-complex carbanion forms."
            },
            {
                "step_index": 4,
                "reaction_example": "Ph(NH2)carbanion.NH3>>PhNH2",
                "note": "Protonation by NH3 restores aromaticity; aniline product."
            }
        ]
    },
    "rt_020": {
        "example_notes": "Benzene + HNO3/H2SO4 gives nitrobenzene (electrophilic aromatic nitration).",
        "current_state_example": ["PhH (benzene)", "HNO3", "H2SO4"],
        "resulting_state_example": ["PhNO2 (nitrobenzene)", "H2O"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "HNO3.H2SO4>>H2NO3+.HSO4-",
                "note": "H2SO4 protonates nitric acid."
            },
            {
                "step_index": 2,
                "reaction_example": "H2NO3+>>NO2+.H2O",
                "note": "Water departs to generate nitronium ion (NO2+)."
            },
            {
                "step_index": 3,
                "reaction_example": "PhH.NO2+>>Ph(H)(NO2)+ sigma complex",
                "note": "Benzene pi system attacks NO2+; Wheland intermediate forms."
            },
            {
                "step_index": 4,
                "reaction_example": "Ph(H)(NO2)+.HSO4->>PhNO2.H2SO4",
                "note": "Deprotonation restores aromaticity; nitrobenzene product."
            }
        ]
    },
    "rt_021": {
        "example_notes": "Dicyclohexylboron enolate of acetaldehyde adds to benzaldehyde; workup gives 3-hydroxy-3-phenylpropanal.",
        "current_state_example": ["boron enolate of acetaldehyde (CH2=CH-OBCy2)", "PhCHO (benzaldehyde)"],
        "resulting_state_example": ["Ph-CH(OH)-CH2-CHO (3-hydroxy-3-phenylpropanal) after workup"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "CH2=CH-OBCy2.PhCHO>>Ph-CH(OBCy2)-CH2-CHO boron aldolate",
                "note": "Enolate alpha-C attacks aldehyde carbonyl; new C-C bond forms; boron aldolate."
            },
            {
                "step_index": 2,
                "reaction_example": "boron aldolate>>Ph-CH(O-BCy2)-CH2-CHO",
                "note": "Alkoxide remains coordinated to boron."
            },
            {
                "step_index": 3,
                "reaction_example": "Ph-CH(O-BCy2)-CH2-CHO.H2O>>Ph-CH(OH)-CH2-CHO.Cy2BOH",
                "note": "Aqueous workup hydrolyzes boron-oxygen bond; beta-hydroxy aldehyde product."
            }
        ]
    },
    "rt_022": {
        "example_notes": "Acetyl chloride + diethylamine + Et3N gives N,N-diethylacetamide.",
        "current_state_example": ["CH3COCl (acetyl chloride)", "HN(C2H5)2 (diethylamine)", "Et3N (base)"],
        "resulting_state_example": ["CH3CON(C2H5)2 (N,N-diethylacetamide)", "Et3N·HCl"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "CH3COCl.HN(Et)2>>CH3C(OH)(NEt2)Cl+ tetrahedral intermediate",
                "note": "Diethylamine attacks acetyl chloride carbonyl; pi bond shifts to O."
            },
            {
                "step_index": 2,
                "reaction_example": "tetrahedral intermediate>>CH3CO-N+H(Et)2.Cl-",
                "note": "Carbonyl reforms; Cl departs to give acylated ammonium."
            },
            {
                "step_index": 3,
                "reaction_example": "CH3CO-N+H(Et)2.Et3N>>CH3CON(Et)2.Et3N-H+",
                "note": "Et3N removes proton; neutral N,N-diethylacetamide product."
            }
        ]
    },
    "rt_023": {
        "example_notes": "Ethanol + acetic anhydride + pyridine gives ethyl acetate.",
        "current_state_example": ["CH3CH2OH (ethanol)", "Ac2O (acetic anhydride)", "pyridine (base)"],
        "resulting_state_example": ["CH3CH2OAc (ethyl acetate)", "AcO-", "pyridine-H+"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "EtOH.Ac2O>>tetrahedral intermediate",
                "note": "Ethanol attacks one anhydride carbonyl."
            },
            {
                "step_index": 2,
                "reaction_example": "tetrahedral intermediate>>EtOAc.AcO-",
                "note": "Collapse reforms carbonyl; acetate departs."
            },
            {
                "step_index": 3,
                "reaction_example": "EtOAc.AcO-.pyridine>>EtOAc.AcOH/pyridine-H+",
                "note": "Pyridine deprotonates the oxonium; neutral ester forms."
            }
        ]
    },
    "rt_024": {
        "example_notes": "1-Propanol + TrCl (triphenylmethyl chloride) + Et3N gives propyl trityl ether.",
        "current_state_example": ["CH3CH2CH2OH (1-propanol)", "Ph3CCl (trityl chloride)", "Et3N"],
        "resulting_state_example": ["CH3CH2CH2-O-CPh3 (propyl trityl ether)", "Cl-", "Et3NH+"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "Ph3CCl>>Ph3C+.Cl-",
                "note": "Trityl chloride ionizes; resonance-stabilized trityl cation forms."
            },
            {
                "step_index": 2,
                "reaction_example": "Ph3C+.PrOH>>PrO(+H)-CPh3",
                "note": "Propanol oxygen attacks trityl cation; oxonium ion forms."
            },
            {
                "step_index": 3,
                "reaction_example": "PrO(+H)-CPh3.Et3N>>PrO-CPh3.Et3NH+",
                "note": "Et3N removes proton; neutral trityl ether product."
            }
        ]
    },
    "rt_025": {
        "example_notes": "1-Propanol + TsCl + Et3N gives n-propyl tosylate.",
        "current_state_example": ["CH3CH2CH2OH (1-propanol)", "TsCl (p-toluenesulfonyl chloride)", "Et3N"],
        "resulting_state_example": ["CH3CH2CH2-OTs (n-propyl tosylate)", "Cl-", "Et3NH+"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "PrOH.TsCl>>sulfurane intermediate",
                "note": "Propanol oxygen attacks sulfonyl S; coordination expands."
            },
            {
                "step_index": 2,
                "reaction_example": "sulfurane>>PrO(+H)-SO2Tol.Cl-",
                "note": "Chloride departs; alkoxy-sulfonyl cation forms."
            },
            {
                "step_index": 3,
                "reaction_example": "PrO(+H)-SO2Tol.Et3N>>PrOTs.Et3NH+",
                "note": "Et3N removes proton; neutral tosylate product."
            }
        ]
    },
    "rt_026": {
        "example_notes": "2-methyl-2-propanol (tert-butanol) + H2SO4 gives 2-methylpropene (isobutylene) by E1.",
        "current_state_example": ["(CH3)3C-OH (tert-butanol)", "H2SO4"],
        "resulting_state_example": ["(CH3)2C=CH2 (isobutylene)", "H2O", "H+"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "(CH3)3COH.H+>>(CH3)3C-OH2+",
                "note": "Protonation of tert-butanol makes water a leaving group."
            },
            {
                "step_index": 2,
                "reaction_example": "(CH3)3C-OH2+>>(CH3)3C+.H2O",
                "note": "Water departs; tertiary carbocation forms."
            },
            {
                "step_index": 3,
                "reaction_example": "(CH3)3C+.HSO4->>(CH3)2C=CH2.H2SO4",
                "note": "Loss of beta-proton gives isobutylene; H+ regenerated."
            }
        ]
    },
    "rt_027": {
        "example_notes": "Ethylene oxide + H2O/H+ gives ethylene glycol (1,2-ethanediol).",
        "current_state_example": ["ethylene oxide (oxirane)", "H2O", "H+"],
        "resulting_state_example": ["HOCH2CH2OH (ethylene glycol)", "H+"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "ethylene oxide.H+>>protonated ethylene oxide",
                "note": "Epoxide oxygen is protonated; ring activated for opening."
            },
            {
                "step_index": 2,
                "reaction_example": "protonated ethylene oxide.H2O>>HO-CH2-CH2-OH2+",
                "note": "Water attacks epoxide carbon; ring opens to give 1,2-diol-oxonium."
            },
            {
                "step_index": 3,
                "reaction_example": "HO-CH2-CH2-OH2+.H2O>>HOCH2CH2OH.H3O+",
                "note": "Deprotonation gives ethylene glycol; H+ regenerated."
            }
        ]
    },
    "rt_028": {
        "example_notes": "Cyclohexanone + Ac2O + Et3N gives 1-acetoxycyclohexene (enol acetate).",
        "current_state_example": ["cyclohexanone", "Ac2O", "Et3N"],
        "resulting_state_example": ["1-acetoxycyclohexene (enol acetate)", "Et3NH+", "AcO-"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "cyclohexanone.Et3N>>cyclohexanone enolate.Et3NH+",
                "note": "Et3N removes alpha-H from cyclohexanone; enolate forms."
            },
            {
                "step_index": 2,
                "reaction_example": "cyclohexanone enolate.Ac2O>>O-acylated enol intermediate",
                "note": "Enolate oxygen attacks acetyl group of Ac2O; O-acylation."
            },
            {
                "step_index": 3,
                "reaction_example": "O-acylated intermediate>>1-acetoxycyclohexene.AcO-",
                "note": "Collapse gives enol acetate; acetate departs."
            }
        ]
    },
    "rt_029": {
        "example_notes": "Cyclohexanone + LDA + TMS-Cl gives 1-(trimethylsilyloxy)cyclohexene (TMS enol ether).",
        "current_state_example": ["cyclohexanone", "LDA (strong base)", "TMSCl (trimethylsilyl chloride)"],
        "resulting_state_example": ["1-(TMSO)-cyclohexene (TMS enol ether)", "LDA-H", "Cl-"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "cyclohexanone.LDA>>cyclohexanone enolate.LDA-H",
                "note": "LDA deprotonates alpha position; lithium enolate forms."
            },
            {
                "step_index": 2,
                "reaction_example": "cyclohexanone enolate.TMSCl>>1-(TMSO)-cyclohexene.Cl-",
                "note": "Enolate O attacks Si (SN2 at Si); chloride departs; TMS enol ether."
            }
        ]
    },
    "rt_030": {
        "example_notes": "Nitromethane + acetone + Et3N gives 1-nitro-2-propanol (beta-nitro alcohol) via Henry reaction.",
        "current_state_example": ["CH3NO2 (nitromethane)", "CH3COCH3 (acetone)", "Et3N"],
        "resulting_state_example": ["CH3CH(OH)CH2NO2 / HOCH2CH(NO2)CH3 (beta-nitro alcohol)"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "CH3NO2.Et3N>>CH2-NO2 nitronate.Et3NH+",
                "note": "Et3N removes acidic CH of nitromethane; nitronate anion forms."
            },
            {
                "step_index": 2,
                "reaction_example": "CH2-NO2.CH3COCH3>>alkoxide nitro adduct",
                "note": "Nitronate C attacks acetone carbonyl; new C-C bond; alkoxide forms."
            },
            {
                "step_index": 3,
                "reaction_example": "alkoxide.H2O>>HOCH2CH(CH3)NO2",
                "note": "Protonation gives beta-nitro alcohol product."
            }
        ]
    },
    "rt_031": {
        "example_notes": "LDA enolate of acetaldehyde adds to benzaldehyde to give 3-hydroxy-3-phenylpropanal.",
        "current_state_example": ["lithium enolate of acetaldehyde (LiO-CH=CH2)", "PhCHO (benzaldehyde)"],
        "resulting_state_example": ["Ph-CH(OH)-CH2-CHO (3-hydroxy-3-phenylpropanal)"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "LiO-CH=CH2.PhCHO>>Ph-CH(OLi)-CH2-CHO aldolate",
                "note": "Enolate alpha-C attacks benzaldehyde; C-C bond forms."
            },
            {
                "step_index": 2,
                "reaction_example": "Ph-CH(OLi)-CH2-CHO>>lithium alkoxide aldol adduct",
                "note": "Product stabilized as lithium alkoxide before workup."
            },
            {
                "step_index": 3,
                "reaction_example": "lithium alkoxide.H2O>>Ph-CH(OH)-CH2-CHO",
                "note": "Aqueous workup protonates alkoxide; beta-hydroxy aldehyde isolated."
            }
        ]
    },
    "rt_032": {
        "example_notes": "Acetylacetone enolate adds to MVK (methyl vinyl ketone) in Michael addition.",
        "current_state_example": ["acetylacetonate anion (nucleophile enolate)", "CH2=CH-CO-CH3 (MVK, enone)"],
        "resulting_state_example": ["CH3CO-CH(CH2CH2COCH3)-COCH3 (1,5-dicarbonyl Michael adduct)"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "acetylacetonate.CH2=CHCOCH3>>conjugate addition enolate",
                "note": "Enolate alpha-C attacks beta-C of MVK (1,4-addition); new C-C bond."
            },
            {
                "step_index": 2,
                "reaction_example": "conjugate addition enolate>>resonance stabilized enolate",
                "note": "Adduct exists as enolate resonance form."
            },
            {
                "step_index": 3,
                "reaction_example": "enolate.H2O>>1,5-dicarbonyl Michael adduct",
                "note": "Protonation gives neutral 1,5-dicarbonyl product."
            }
        ]
    },
    "rt_033": {
        "example_notes": "Ethyl chloroacetate + benzaldehyde + NaOEt gives ethyl 2,3-epoxyphenylpropanoate (Darzens ester).",
        "current_state_example": ["ClCH2CO2Et (ethyl chloroacetate)", "PhCHO (benzaldehyde)", "NaOEt"],
        "resulting_state_example": ["Ph-CH(-O-)CH-CO2Et (ethyl 2,3-epoxy-3-phenylpropanoate)", "Cl-", "EtOH"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "ClCH2CO2Et.NaOEt>>Cl-CH-CO2Et enolate.EtOH",
                "note": "Base removes alpha-H from ethyl chloroacetate; halo ester enolate."
            },
            {
                "step_index": 2,
                "reaction_example": "ClCH-CO2Et.PhCHO>>Ph-CH(O-)-CHCl-CO2Et",
                "note": "Enolate attacks benzaldehyde; beta-halo alkoxide forms."
            },
            {
                "step_index": 3,
                "reaction_example": "Ph-CH(O-)-CHCl-CO2Et>>Ph-CH(-O-)CH-CO2Et.Cl-",
                "note": "Alkoxide displaces Cl intramolecularly; glycidic ester epoxide forms."
            }
        ]
    },
    "rt_034": {
        "example_notes": "Ethyl acetate + NaOEt + ethyl acetate gives ethyl acetoacetate (Claisen condensation).",
        "current_state_example": ["CH3CO2Et (ethyl acetate, donor)", "CH3CO2Et (ethyl acetate, acceptor)", "NaOEt"],
        "resulting_state_example": ["CH3COCH2CO2Et (ethyl acetoacetate) as enolate / after protonation"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "CH3CO2Et.NaOEt>>CH2-CO2Et enolate.EtOH",
                "note": "NaOEt deprotonates alpha-H of ethyl acetate; ester enolate."
            },
            {
                "step_index": 2,
                "reaction_example": "CH2-CO2Et.CH3CO2Et>>tetrahedral Claisen intermediate",
                "note": "Enolate attacks second ester's carbonyl; tetrahedral intermediate."
            },
            {
                "step_index": 3,
                "reaction_example": "tetrahedral intermediate>>CH3COCH2CO2Et enolate.EtO-",
                "note": "Ethoxide departs; ethyl acetoacetate product (as enolate) forms."
            }
        ]
    },
    "rt_035": {
        "example_notes": "LDA enolate of acetone reacts with dimethyl disulfide (MeSSMe) to give alpha-(methylthio)acetone.",
        "current_state_example": ["LDA enolate of acetone (CH3-CO-CH2-)", "CH3-S-S-CH3 (dimethyl disulfide)"],
        "resulting_state_example": ["CH3-CO-CH2-SCH3 (alpha-methylthio acetone)", "CH3S-"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "acetone.LDA>>CH3COCH2- enolate.LDA-H",
                "note": "LDA deprotonates alpha position of acetone."
            },
            {
                "step_index": 2,
                "reaction_example": "CH3COCH2-.CH3SSCH3>>CH3COCH2SCH3.CH3S-",
                "note": "Enolate C attacks one S of disulfide; S-S bond cleaves; thiolate departs."
            },
            {
                "step_index": 3,
                "reaction_example": "CH3COCH2SCH3 enolate.H2O>>CH3COCH2SCH3",
                "note": "Protonation gives neutral alpha-thioketone."
            }
        ]
    },
    "rt_036": {
        "example_notes": "Corey-Chaykovsky: trimethylsulfonium ylide + benzaldehyde gives styrene oxide.",
        "current_state_example": ["trimethylsulfonium methylide (Me3S+=CH2) precursor", "PhCHO (benzaldehyde)", "NaH (base)"],
        "resulting_state_example": ["Ph-CH(-O-)CH2 (styrene oxide)", "dimethylsulfide (Me2S)"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "Me3S+CH3 (TMS iodide).NaH>>Me3S+=CH2 (sulfonium ylide).H2",
                "note": "NaH generates trimethylsulfonium methylide from trimethylsulfonium salt."
            },
            {
                "step_index": 2,
                "reaction_example": "Me3S+=CH2.PhCHO>>Ph-CH(O-)-CH2-S+Me2 betaine",
                "note": "Ylide C attacks benzaldehyde C; betaine intermediate."
            },
            {
                "step_index": 3,
                "reaction_example": "betaine>>Ph-CH(-O-)CH2.Me2S",
                "note": "Alkoxide displaces Me2S intramolecularly; styrene oxide forms."
            }
        ]
    },
    "rt_037": {
        "example_notes": "Pinacol (2,3-dimethyl-2,3-butanediol) + H2SO4 gives pinacolone (3,3-dimethylbutan-2-one).",
        "current_state_example": ["(CH3)2C(OH)-C(OH)(CH3)2 (pinacol, 2,3-dimethyl-2,3-butanediol)", "H2SO4"],
        "resulting_state_example": ["(CH3)3C-CO-CH3 (pinacolone)", "H2O", "H+"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "pinacol.H+>>protonated pinacol",
                "note": "One hydroxyl is protonated by H2SO4."
            },
            {
                "step_index": 2,
                "reaction_example": "protonated pinacol>>rearranged oxocarbenium.H2O",
                "note": "Water leaves; 1,2-methyl migration gives oxocarbenium (pinacolone cation)."
            },
            {
                "step_index": 3,
                "reaction_example": "oxocarbenium.H2O>>pinacolone.H+",
                "note": "Deprotonation gives neutral pinacolone; H+ regenerated."
            }
        ]
    },
    "rt_038": {
        "example_notes": "1-propanol + conc. HCl gives 1-chloropropane via SN1/SN2.",
        "current_state_example": ["CH3CH2CH2OH (1-propanol)", "HCl"],
        "resulting_state_example": ["CH3CH2CH2Cl (1-chloropropane)", "H2O"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "PrOH.HCl>>PrOH2+.Cl-",
                "note": "Protonation of propanol activates it."
            },
            {
                "step_index": 2,
                "reaction_example": "PrOH2+>>Pr+.H2O",
                "note": "Water departs; carbocation (or highly activated substrate) forms."
            },
            {
                "step_index": 3,
                "reaction_example": "Pr+.Cl->>PrCl",
                "note": "Chloride attacks cation; 1-chloropropane product."
            }
        ]
    },
    "rt_039": {
        "example_notes": "Acetone + Br2/AcOH gives bromoacetone (alpha-bromination).",
        "current_state_example": ["CH3COCH3 (acetone)", "Br2", "AcOH"],
        "resulting_state_example": ["BrCH2COCH3 (bromoacetone)", "HBr"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "CH3COCH3.AcOH>>CH3CO-CH2 enol / enolate",
                "note": "Acid-catalyzed enolization of acetone."
            },
            {
                "step_index": 2,
                "reaction_example": "CH3CO-CH2 enol.Br2>>alpha-bromo oxonium.Br-",
                "note": "Enol pi bond attacks Br2; alpha-bromo oxonium and Br- form."
            },
            {
                "step_index": 3,
                "reaction_example": "alpha-bromo oxonium>>BrCH2COCH3 precursor",
                "note": "Carbonyl restored by tautomerization."
            },
            {
                "step_index": 4,
                "reaction_example": "BrCH2COCH3 precursor.AcOH>>BrCH2COCH3.HBr",
                "note": "Neutral bromoacetone isolated; HBr by-product."
            }
        ]
    },
    "rt_040": {
        "example_notes": "Dimethylamine + formaldehyde + NaBH3CN (Eschweiler-Clarke or reductive methylation) gives trimethylamine.",
        "current_state_example": ["(CH3)2NH (dimethylamine)", "HCHO (formaldehyde)", "NaBH3CN", "buffer pH 7"],
        "resulting_state_example": ["(CH3)3N (trimethylamine)"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "(CH3)2NH.HCHO>>carbinolamine (CH3)2N-CH2-OH",
                "note": "Dimethylamine attacks formaldehyde; carbinolamine adduct."
            },
            {
                "step_index": 2,
                "reaction_example": "carbinolamine>>protonated carbinolamine",
                "note": "Proton transfer at O/N prepares for dehydration."
            },
            {
                "step_index": 3,
                "reaction_example": "protonated carbinolamine>>(CH3)2N+=CH2 iminium.H2O",
                "note": "Water eliminated; iminium ion (CH3)2N+=CH2 forms."
            },
            {
                "step_index": 4,
                "reaction_example": "(CH3)2N+=CH2.NaBH3CN>>(CH3)3N+H ammonium",
                "note": "Hydride from NaBH3CN reduces iminium C; trimethylammonium forms."
            },
            {
                "step_index": 5,
                "reaction_example": "(CH3)3NH+.base>>(CH3)3N",
                "note": "Deprotonation gives neutral trimethylamine."
            }
        ]
    },
    "rt_041": {
        "example_notes": "1,3-dimethylimidazolium salt + KOtBu gives 1,3-dimethylimidazol-2-ylidene (NHC carbene).",
        "current_state_example": ["1,3-dimethylimidazolium iodide (azolium salt)", "KOtBu (strong base)"],
        "resulting_state_example": ["1,3-dimethylimidazol-2-ylidene (NHC)", "KI", "tBuOH"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "1,3-dimethylimidazolium.KOtBu>>1,3-dimethylimidazol-2-ylidene.KI.tBuOH",
                "note": "KOtBu removes acidic C2-H; electron pair remains on C2 to give NHC carbene."
            }
        ]
    },
    "rt_042": {
        "example_notes": "Sodium ethoxide + n-propyl tosylate gives n-propyl ethyl ether by SN2.",
        "current_state_example": ["NaOC2H5 (sodium ethoxide)", "CH3CH2CH2-OTs (n-propyl tosylate)"],
        "resulting_state_example": ["CH3CH2-O-CH2CH2CH3 (ethyl propyl ether)", "TsO-"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "EtO-.PrOTs>>EtO-Pr.TsO-",
                "note": "Ethoxide attacks primary carbon; tosylate departs (SN2). Ether forms."
            }
        ]
    },
    "rt_043": {
        "example_notes": "NaN3 displaces mesylate from methyl mesylate to give methyl azide.",
        "current_state_example": ["NaN3 (sodium azide)", "CH3-OMs (methyl mesylate, activated derivative)"],
        "resulting_state_example": ["CH3-N3 (methyl azide)"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "N3-.CH3OMs>>CH3-N3.MsO-",
                "note": "Azide attacks methyl carbon; mesylate departs (SN2)."
            }
        ]
    },
    "rt_044": {
        "example_notes": "Lithium enolate of acetone + methyl iodide gives methyl isopropenyl ether (O-methylation).",
        "current_state_example": ["LDA enolate of acetone (O-enolate)", "CH3I (methyl iodide)"],
        "resulting_state_example": ["CH3-O-C(CH3)=CH2 (methyl isopropenyl ether)"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "CH3COCH3.LDA>>LiO-C(CH3)=CH2 enolate.LDA-H",
                "note": "LDA deprotonates acetone alpha position; lithium enolate (O-form)."
            },
            {
                "step_index": 2,
                "reaction_example": "LiO-C(CH3)=CH2.CH3I>>CH3O-C(CH3)=CH2.LiI",
                "note": "Enolate O attacks CH3I (SN2); iodide departs; methyl enol ether."
            }
        ]
    },
    "rt_045": {
        "example_notes": "PPh3 + benzyl bromide gives benzyltriphenylphosphonium bromide (Wittig precursor).",
        "current_state_example": ["PPh3 (triphenylphosphine)", "Ph-CH2-Br (benzyl bromide)"],
        "resulting_state_example": ["[Ph-CH2-PPh3]+ Br- (benzyltriphenylphosphonium bromide)"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "PPh3.PhCH2Br>>[PhCH2PPh3]+ Br-",
                "note": "Phosphine lone pair attacks benzyl carbon; bromide departs (SN2). Phosphonium salt."
            }
        ]
    },
    "rt_046": {
        "example_notes": "Propene + HBr gives 2-bromopropane (Markovnikov addition).",
        "current_state_example": ["CH3CH=CH2 (propene)", "HBr"],
        "resulting_state_example": ["CH3CHBrCH3 (2-bromopropane)"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "CH3CH=CH2.HBr>>CH3CH+(CH3).Br-",
                "note": "Proton adds to less substituted alkene carbon; secondary carbocation forms (Markovnikov)."
            },
            {
                "step_index": 2,
                "reaction_example": "CH3CH+(CH3).Br->>CH3CHBrCH3",
                "note": "Bromide captures secondary carbocation; 2-bromopropane product."
            }
        ]
    },
    "rt_047": {
        "example_notes": "Propene + HI gives 2-iodopropane (Markovnikov addition).",
        "current_state_example": ["CH3CH=CH2 (propene)", "HI"],
        "resulting_state_example": ["CH3CHICH3 (2-iodopropane)"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "CH3CH=CH2.HI>>CH3CH+(CH3).I-",
                "note": "Proton adds to terminal carbon; secondary carbocation forms."
            },
            {
                "step_index": 2,
                "reaction_example": "CH3CH+(CH3).I->>CH3CHICH3",
                "note": "Iodide attacks carbocation; 2-iodopropane product."
            }
        ]
    },
    "rt_048": {
        "example_notes": "Propylene oxide + HBr gives 1-bromo-2-propanol (halohydrin; SN2 at more substituted C in acid).",
        "current_state_example": ["propylene oxide (1,2-epoxypropane)", "HBr"],
        "resulting_state_example": ["BrCH2-CH(OH)-CH3 (1-bromo-2-propanol)"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "propylene oxide.HBr>>protonated propylene oxide.Br-",
                "note": "Epoxide oxygen protonated by HBr."
            },
            {
                "step_index": 2,
                "reaction_example": "protonated propylene oxide.Br->>BrCH2CH(OH)CH3",
                "note": "Bromide attacks more substituted C (acid conditions, SN1-like); ring opens; halohydrin forms."
            }
        ]
    },
    "rt_049": {
        "example_notes": "Benzene + Br2/FeBr3 gives bromobenzene (electrophilic aromatic substitution).",
        "current_state_example": ["PhH (benzene)", "Br2", "FeBr3 (Lewis acid)"],
        "resulting_state_example": ["PhBr (bromobenzene)", "HBr"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "Br2.FeBr3>>Br+FeBr4- (activated bromine electrophile)",
                "note": "FeBr3 coordinates to Br2 making Br more electrophilic."
            },
            {
                "step_index": 2,
                "reaction_example": "PhH.Br+>>Ph(H)(Br)+ sigma complex",
                "note": "Ring attacks electrophilic Br; Wheland sigma complex (arenium ion)."
            },
            {
                "step_index": 3,
                "reaction_example": "Ph(H)(Br)+.FeBr4->>PhBr.HBr.FeBr3",
                "note": "Deprotonation restores aromaticity; bromobenzene product."
            }
        ]
    },
    "rt_050": {
        "example_notes": "Benzaldehyde + HCN gives mandelonitrile (PhCH(OH)CN) — cyanohydrin.",
        "current_state_example": ["PhCHO (benzaldehyde)", "HCN (from NaCN + H+)"],
        "resulting_state_example": ["PhCH(OH)CN (mandelonitrile)"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "PhCHO.CN->>PhCH(O-)CN alkoxide",
                "note": "Cyanide attacks benzaldehyde carbonyl; alkoxide forms."
            },
            {
                "step_index": 2,
                "reaction_example": "PhCH(O-)CN.H+>>PhCH(OH)CN",
                "note": "Alkoxide protonated; mandelonitrile product."
            }
        ]
    },
    "rt_051": {
        "example_notes": "Methyl vinyl ketone (MVK) + HCN gives 5-oxohexanenitrile (4-cyano-2-butanone).",
        "current_state_example": ["CH2=CH-CO-CH3 (MVK, methyl vinyl ketone)", "HCN"],
        "resulting_state_example": ["NC-CH2-CH2-CO-CH3 (4-oxopentanenitrile)"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "CH2=CHCOCH3.CN->>NC-CH2-CH2-CO-CH3 enolate",
                "note": "Cyanide adds 1,4 to beta-carbon of MVK; enolate at alpha of ketone."
            },
            {
                "step_index": 2,
                "reaction_example": "enolate.H+>>NC-CH2CH2COCH3",
                "note": "Protonation gives neutral beta-cyano ketone."
            }
        ]
    },
    "rt_052": {
        "example_notes": "MVK (methyl vinyl ketone) + HBr gives 4-bromo-2-butanone (1,4-addition of HBr to enone).",
        "current_state_example": ["CH2=CH-CO-CH3 (MVK)", "HBr"],
        "resulting_state_example": ["Br-CH2-CH2-CO-CH3 (4-bromo-2-butanone, beta-halo ketone)"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "CH2=CHCOCH3.HBr>>CH3CO-CH2-CH2+ cation.Br-",
                "note": "Protonation at alpha carbon of MVK; resonance-stabilized cation; bromide released."
            },
            {
                "step_index": 2,
                "reaction_example": "CH3COCH2CH2+.Br->>BrCH2CH2COCH3",
                "note": "Bromide attacks beta carbon; 4-bromo-2-butanone product."
            }
        ]
    },
    "rt_053": {
        "example_notes": "Cinnamaldehyde (PhCH=CH-CHO) + DIBAL-H (or Red-Al) gives cinnamyl alcohol (1,2-selective).",
        "current_state_example": ["PhCH=CH-CHO (cinnamaldehyde, enal)", "DIBAL-H (bulky hydride)"],
        "resulting_state_example": ["PhCH=CH-CH2OH (cinnamyl alcohol, allylic alcohol)"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "PhCH=CHCHO.DIBAL-H>>PhCH=CH-CH(O-AlR2) alkoxide",
                "note": "Hydride delivers to aldehyde C=O (1,2-selective); allylic alkoxide; alkene intact."
            },
            {
                "step_index": 2,
                "reaction_example": "alkoxide.H2O>>PhCH=CHCH2OH",
                "note": "Aqueous workup protonates alkoxide; cinnamyl alcohol."
            }
        ]
    },
    "rt_054": {
        "example_notes": "Crotonaldehyde (CH3CH=CHCHO) + MeMgBr gives 1-methylbut-2-en-1-ol (1,2-addition).",
        "current_state_example": ["CH3CH=CH-CHO (crotonaldehyde, enal)", "CH3MgBr (methylmagnesium bromide)"],
        "resulting_state_example": ["CH3CH=CH-CH(OH)CH3 (1-methylbut-2-en-1-ol, allylic alcohol)"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "CH3CH=CHCHO.CH3MgBr>>CH3CH=CH-CH(OMgBr)CH3 alkoxide",
                "note": "Grignard methyl attacks enal aldehyde carbonyl (1,2-addition); allylic alkoxide."
            },
            {
                "step_index": 2,
                "reaction_example": "alkoxide.H2O>>CH3CH=CHCH(OH)CH3",
                "note": "Workup protonates alkoxide; allylic alcohol product."
            }
        ]
    },
    "rt_055": {
        "example_notes": "4-nitrochlorobenzene + NaOH gives 4-nitrophenol by SNAr.",
        "current_state_example": ["4-O2N-C6H4-Cl (4-nitrochlorobenzene)", "NaOH"],
        "resulting_state_example": ["4-O2N-C6H4-OH (4-nitrophenol)", "Cl-"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "4-NO2-C6H4-Cl.OH->>Meisenheimer complex (4-NO2-C6H4(OH)(Cl)-)",
                "note": "Hydroxide adds to activated carbon bearing Cl; anionic Meisenheimer complex."
            },
            {
                "step_index": 2,
                "reaction_example": "Meisenheimer complex>>4-NO2-C6H4-OH.Cl-",
                "note": "Chloride expelled; aromaticity restored; 4-nitrophenol product."
            }
        ]
    },
    "rt_056": {
        "example_notes": "Anisole (PhOCH3) + HBr gives phenol + bromomethane (ether cleavage).",
        "current_state_example": ["PhOCH3 (anisole, aryl methyl ether)", "HBr"],
        "resulting_state_example": ["PhOH (phenol)", "CH3Br (bromomethane)"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "PhOCH3.HBr>>PhO+(H)CH3.Br-",
                "note": "Ether oxygen protonated by HBr; oxonium intermediate."
            },
            {
                "step_index": 2,
                "reaction_example": "PhO+(H)CH3.Br->>PhOH.CH3Br",
                "note": "Bromide attacks methyl (SN2); C-O bond to phenol breaks; phenol + bromomethane."
            }
        ]
    },
    "rt_057": {
        "example_notes": "Acetonitrile (CH3CN) + NaH + methyl iodide gives propionitrile (C2H5CN).",
        "current_state_example": ["CH3CN (acetonitrile)", "NaH", "CH3I (methyl iodide)"],
        "resulting_state_example": ["CH3CH2CN (propionitrile)"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "CH3CN.NaH>>CH2-CN anion.H2",
                "note": "NaH removes alpha-H from acetonitrile; nitrile-stabilized carbanion."
            },
            {
                "step_index": 2,
                "reaction_example": "CH2-CN.CH3I>>CH3CH2CN.I-",
                "note": "Carbanion attacks methyl iodide (SN2); iodide departs; propionitrile."
            }
        ]
    },
    "rt_058": {
        "example_notes": "3-chloropropanenitrile + KOtBu gives cyclopropanecarbonitrile via intramolecular ring closure.",
        "current_state_example": ["Cl-CH2-CH2-CH2-CN (3-chloropropanenitrile)", "KOtBu"],
        "resulting_state_example": ["cyclopropane-CN (cyclopropanecarbonitrile)"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "ClCH2CH2CH2CN.KOtBu>>ClCH2CH2CH-CN anion.tBuOH",
                "note": "KOtBu deprotonates alpha-C; nitrile-stabilized carbanion forms."
            },
            {
                "step_index": 2,
                "reaction_example": "ClCH2CH2CH-CN>>cyclopropane-CN.Cl-",
                "note": "Carbanion attacks tethered CH2Cl (SN2, 3-exo-tet); ring closes; Cl- departs."
            }
        ]
    },
    "rt_059": {
        "example_notes": "4-chloronitrobutane + K2CO3 gives nitrocyclopropane via intramolecular nitronate cyclization.",
        "current_state_example": ["O2N-CH2-CH2-CH2-Cl (3-chloropropyl nitro compound)", "K2CO3"],
        "resulting_state_example": ["nitrocyclopropane (1-nitrocyclopropane)"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "O2N-CH2-CH2CH2Cl.K2CO3>>nitronate anion",
                "note": "Base deprotonates alpha to nitro; nitronate/carbanion."
            },
            {
                "step_index": 2,
                "reaction_example": "nitronate>>nitrocyclopropane.Cl-",
                "note": "Carbanion attacks tethered CH2Cl (SN2); 3-membered ring; Cl- leaves."
            }
        ]
    },
    "rt_060": {
        "example_notes": "Acetylacetone + K2CO3 + ethyl iodide gives 3-ethylacetylacetone.",
        "current_state_example": ["CH3COCH2COCH3 (acetylacetone)", "K2CO3", "C2H5I (ethyl iodide)"],
        "resulting_state_example": ["CH3CO-CH(C2H5)-COCH3 (3-ethylacetylacetone)"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "CH3COCH2COCH3.K2CO3>>CH3COCHCOCH3 enolate",
                "note": "K2CO3 deprotonates acidic methylene of acetylacetone."
            },
            {
                "step_index": 2,
                "reaction_example": "CH3COCHCOCH3.C2H5I>>CH3COCH(C2H5)COCH3.I-",
                "note": "Enolate C attacks ethyl iodide (SN2); iodide departs; alkylated product."
            }
        ]
    },
    "rt_061": {
        "example_notes": "Diethyl malonate + NaOEt + n-butyl bromide gives diethyl 2-butylmalonate.",
        "current_state_example": ["(EtO2C)2CH2 (diethyl malonate)", "NaOEt", "n-BuBr (n-butyl bromide)"],
        "resulting_state_example": ["(EtO2C)2CH-nBu (diethyl n-butylmalonate)"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "(EtO2C)2CH2.NaOEt>>(EtO2C)2CH- malonate enolate.EtOH",
                "note": "NaOEt deprotonates malonate methylene; stabilized enolate."
            },
            {
                "step_index": 2,
                "reaction_example": "(EtO2C)2CH-.n-BuBr>>(EtO2C)2CH-nBu.Br-",
                "note": "Malonate enolate attacks n-BuBr (SN2); bromide leaves."
            }
        ]
    },
    "rt_062": {
        "example_notes": "Ethyl acetoacetate + NaOEt + methyl iodide gives ethyl 2-methyl-3-oxobutanoate.",
        "current_state_example": ["CH3COCH2CO2Et (ethyl acetoacetate)", "NaOEt", "CH3I"],
        "resulting_state_example": ["CH3COCH(CH3)CO2Et (ethyl 2-methylacetoacetate)"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "CH3COCH2CO2Et.NaOEt>>CH3COCHCO2Et enolate.EtOH",
                "note": "NaOEt deprotonates acidic alpha-H; stabilized beta-keto ester enolate."
            },
            {
                "step_index": 2,
                "reaction_example": "CH3COCHCO2Et.CH3I>>CH3COCH(CH3)CO2Et.I-",
                "note": "Enolate C attacks CH3I (SN2); methylated product."
            }
        ]
    },
    "rt_063": {
        "example_notes": "3-hydroxypropanal + NaOH gives acrolein (propenal, CH2=CHCHO) by aldol dehydration.",
        "current_state_example": ["HOCH2CH2CHO (3-hydroxypropanal, beta-hydroxy aldehyde)", "NaOH"],
        "resulting_state_example": ["CH2=CH-CHO (acrolein, propenal)"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "HOCH2CH2CHO.NaOH>>alkoxide or enolate intermediate",
                "note": "NaOH deprotonates the alpha position or activates the beta-OH for elimination."
            },
            {
                "step_index": 2,
                "reaction_example": "activated intermediate>>CH2=CHCHO.H2O",
                "note": "Elimination of beta-OH gives acrolein; water expelled."
            }
        ]
    },
    "rt_064": {
        "example_notes": "Dimethylsulfonium methylide + cyclohex-2-en-1-one gives bicyclo[4.1.0] ketone (cyclopropanation of enone).",
        "current_state_example": ["Me2S+=CH2 (dimethylsulfonium methylide)", "cyclohex-2-en-1-one"],
        "resulting_state_example": ["bicyclo[4.1.0]heptan-2-one (cyclopropyl ketone)"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "Me2S+=CH2.cyclohexenone>>betaine adduct",
                "note": "Ylide C attacks beta-C of cyclohexenone; betaine intermediate with carbanion."
            },
            {
                "step_index": 2,
                "reaction_example": "betaine>>bicyclo[4.1.0]heptan-2-one.Me2S",
                "note": "Carbanion closes cyclopropane ring; Me2S expelled."
            }
        ]
    },
    "rt_065": {
        "example_notes": "Trimethylsilylacetylene (TMS-C#C-H) + TBAF (tetrabutylammonium fluoride) gives terminal acetylene.",
        "current_state_example": ["HC#C-Si(CH3)3 (TMS-acetylene)", "TBAF (fluoride source)"],
        "resulting_state_example": ["HC#CH (acetylene)"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "HC#C-TMS.F->>HC#CH.TMSF",
                "note": "Fluoride attacks Si (Si-F bond very strong); Si-C bond breaks; terminal alkyne deprotected."
            }
        ]
    },
    "rt_066": {
        "example_notes": "NaSMe (sodium methanethiolate) + 2-chloropyridine gives 2-(methylthio)pyridine by SNAr.",
        "current_state_example": ["NaSCH3 (sodium methanethiolate)", "2-chloropyridine"],
        "resulting_state_example": ["2-(CH3S)-pyridine (2-(methylthio)pyridine)", "Cl-"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "2-Cl-pyridine.MeS->>Meisenheimer complex",
                "note": "Methanethiolate adds to C-2 of 2-chloropyridine; anionic Meisenheimer complex."
            },
            {
                "step_index": 2,
                "reaction_example": "Meisenheimer complex>>2-(MeS)-pyridine.Cl-",
                "note": "Chloride expelled; aromaticity restored; 2-(methylthio)pyridine."
            }
        ]
    },
    "rt_067": {
        "example_notes": "Pyridine N-oxide + PCl3 gives pyridine (deoxygenation).",
        "current_state_example": ["pyridine N-oxide", "PCl3"],
        "resulting_state_example": ["pyridine", "O=PCl3 (phosphoryl chloride)"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "pyridine-N-oxide.PCl3>>O-P adduct",
                "note": "N-oxide O attacks P of PCl3; oxygen-phosphorus bond forms."
            },
            {
                "step_index": 2,
                "reaction_example": "O-P adduct>>pyridine.OPCl3",
                "note": "N-O bond breaks; oxygen transferred to P; pyridine deoxygenated."
            }
        ]
    },
    "rt_068": {
        "example_notes": "2-bromothiophene + n-BuLi gives 2-thienyllithium by halogen-metal exchange.",
        "current_state_example": ["2-bromothiophene", "n-BuLi (n-butyllithium)"],
        "resulting_state_example": ["2-thienyllithium (2-lithiothiophene)", "n-BuBr"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "2-Br-thiophene.n-BuLi>>2-Li-thiophene.n-BuBr",
                "note": "n-BuLi exchanges Li for Br; 2-thienyllithium + n-BuBr. Very fast exchange."
            }
        ]
    },
    "rt_069": {
        "example_notes": "Phenoxide + 2-chloropyrimidine gives 2-phenoxypyrimidine by SNAr.",
        "current_state_example": ["PhO- (sodium phenoxide)", "2-chloropyrimidine"],
        "resulting_state_example": ["2-phenoxypyrimidine", "Cl-"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "2-Cl-pyrimidine.PhO->>Meisenheimer complex",
                "note": "Phenoxide attacks C-2 of 2-chloropyrimidine; anionic sigma complex."
            },
            {
                "step_index": 2,
                "reaction_example": "Meisenheimer complex>>2-phenoxypyrimidine.Cl-",
                "note": "Chloride expelled; aromaticity restored; aryl ether product."
            }
        ]
    },
    "rt_070": {
        "example_notes": "Pyrazole anion + methyl tosylate gives 1-methylpyrazole (N-methylation).",
        "current_state_example": ["pyrazolide anion (deprotonated pyrazole)", "CH3OTs (methyl tosylate)"],
        "resulting_state_example": ["1-methylpyrazole (N-methyl pyrazole)"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "pyrazolide.CH3OTs>>1-methylpyrazole.OTs-",
                "note": "Ring N attacks methyl tosylate (SN2); tosylate departs; N-methylpyrazole."
            }
        ]
    },
    "rt_071": {
        "example_notes": "2-chloroethanol + NaOH gives ethylene oxide (chlorohydrin cyclization).",
        "current_state_example": ["ClCH2CH2OH (2-chloroethanol)", "NaOH"],
        "resulting_state_example": ["ethylene oxide (oxirane)", "Cl-", "H2O"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "ClCH2CH2OH.NaOH>>ClCH2CH2O- alkoxide.H2O",
                "note": "NaOH deprotonates 2-chloroethanol; alkoxide nucleophile."
            },
            {
                "step_index": 2,
                "reaction_example": "ClCH2CH2O->>ethylene oxide.Cl-",
                "note": "Alkoxide attacks adjacent CH2Cl intramolecularly (SN2); epoxide ring closes."
            }
        ]
    },
    "rt_072": {
        "example_notes": "KSAc (potassium thioacetate) + cinnamyl bromide gives cinnamyl thioacetate.",
        "current_state_example": ["KSAc (potassium thioacetate, CH3COS-)", "Ph-CH=CH-CH2Br (cinnamyl bromide)"],
        "resulting_state_example": ["Ph-CH=CH-CH2-SAc (cinnamyl thioacetate)", "Br-"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "CH3COS-.PhCH=CHCH2Br>>PhCH=CHCH2-SAc.Br-",
                "note": "Thioacetate S attacks activated allylic/benzylic carbon; bromide departs (SN2)."
            }
        ]
    },
    "rt_073": {
        "example_notes": "PhCH2-SAc (benzyl thioacetate) + NaOMe gives benzyl thiolate (PhCH2S-).",
        "current_state_example": ["PhCH2-S-COCH3 (benzyl thioacetate)", "NaOMe (base)"],
        "resulting_state_example": ["PhCH2-S- (benzyl thiolate)", "CH3CO2Me (methyl acetate byproduct)"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "PhCH2SAc.NaOMe>>PhCH2S-.MeO-COCH3",
                "note": "Methoxide attacks thioacetyl carbonyl; thiolate departs; acyl protecting group cleaved."
            }
        ]
    },
    "rt_074": {
        "example_notes": "Benzene + oleum (SO3/H2SO4) gives benzenesulfonic acid (arene sulfonation).",
        "current_state_example": ["PhH (benzene)", "SO3/H2SO4 (oleum)"],
        "resulting_state_example": ["PhSO3H (benzenesulfonic acid)"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "SO3/H2SO4>>sulfonating electrophile (HS2O7- / SO3)",
                "note": "Acid medium activates SO3 as electrophilic sulfonating species."
            },
            {
                "step_index": 2,
                "reaction_example": "PhH.SO3>>Ph(H)(SO3H)+ sigma complex",
                "note": "Benzene ring attacks electrophilic S; arenium sigma complex."
            },
            {
                "step_index": 3,
                "reaction_example": "sigma complex.HSO4->>PhSO3H.H2SO4",
                "note": "Proton removed; aromaticity restored; benzenesulfonic acid."
            }
        ]
    },
    "rt_075": {
        "example_notes": "Benzene + 2-methylpropan-2-ol (tBuOH) + H3PO4 gives tert-butylbenzene (Friedel-Crafts alkylation).",
        "current_state_example": ["PhH (benzene)", "(CH3)3COH (tert-butanol)", "H3PO4"],
        "resulting_state_example": ["Ph-C(CH3)3 (tert-butylbenzene)"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "(CH3)3COH.H+>>(CH3)3C+.H2O",
                "note": "Acid protonates tert-butanol; tert-butyl carbocation forms."
            },
            {
                "step_index": 2,
                "reaction_example": "PhH.(CH3)3C+>>Ph(H)(tBu)+ sigma complex",
                "note": "Ring attacks tert-butyl carbocation; Wheland intermediate."
            },
            {
                "step_index": 3,
                "reaction_example": "sigma complex.H+>>Ph-tBu.H+",
                "note": "Rearomatization via deprotonation; tert-butylbenzene product."
            }
        ]
    },
    "rt_076": {
        "example_notes": "Cyclohexanone + mCPBA gives caprolactone (epsilon-caprolactone, 6-membered lactone) by Baeyer-Villiger.",
        "current_state_example": ["cyclohexanone", "mCPBA (meta-chloroperoxybenzoic acid)"],
        "resulting_state_example": ["epsilon-caprolactone (oxepan-2-one)", "mCBA (meta-chlorobenzoic acid)"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "cyclohexanone.mCPBA>>Criegee intermediate (tetrahedral adduct)",
                "note": "Peroxy oxygen of mCPBA attacks cyclohexanone carbonyl; tetrahedral Criegee intermediate."
            },
            {
                "step_index": 2,
                "reaction_example": "Criegee intermediate>>epsilon-caprolactone.mCBA",
                "note": "1,2-alkyl migration (alpha-C migrates); O-O bond cleaves; lactone + mCBA."
            }
        ]
    },
    "rt_077": {
        "example_notes": "1-butanol + PBr3 gives 1-bromobutane.",
        "current_state_example": ["CH3CH2CH2CH2OH (1-butanol)", "PBr3"],
        "resulting_state_example": ["CH3CH2CH2CH2Br (1-bromobutane)"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "BuOH.PBr3>>Bu-O-PBr2 alkoxyphosphorus.HBr",
                "note": "Alcohol O attacks P; activates C-O bond as good leaving group."
            },
            {
                "step_index": 2,
                "reaction_example": "Bu-O-PBr2.Br->>BuBr.OPBr2-",
                "note": "Bromide attacks butyl C (SN2); C-O breaks; 1-bromobutane product."
            }
        ]
    },
    "rt_078": {
        "example_notes": "Acetaldehyde (CH3CHO) + NH2OH gives acetaldehyde oxime (CH3CH=NOH).",
        "current_state_example": ["CH3CHO (acetaldehyde)", "NH2OH (hydroxylamine)"],
        "resulting_state_example": ["CH3CH=N-OH (acetaldehyde oxime)"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "CH3CHO.NH2OH>>CH3CH(OH)(NHOH) hemiaminal",
                "note": "Hydroxylamine N attacks acetaldehyde; hemiaminal forms."
            },
            {
                "step_index": 2,
                "reaction_example": "hemiaminal>>proton-transfer intermediate",
                "note": "Proton transfers activate OH for departure."
            },
            {
                "step_index": 3,
                "reaction_example": "intermediate>>CH3CH=NOH.H2O",
                "note": "Water eliminated; C=N bond forms; acetaldehyde oxime product."
            }
        ]
    },
    "rt_079": {
        "example_notes": "Cyclohexanone + morpholine + NaBH(OAc)3 gives N-cyclohexylmorpholine (reductive amination).",
        "current_state_example": ["cyclohexanone", "morpholine (HN(CH2CH2)2O)", "NaBH(OAc)3"],
        "resulting_state_example": ["N-cyclohexylmorpholine (tertiary amine)"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "cyclohexanone.morpholine>>hemiaminal (N-hydroxyl adduct)",
                "note": "Morpholine N attacks cyclohexanone; hemiaminal."
            },
            {
                "step_index": 2,
                "reaction_example": "hemiaminal>>cyclohexylidene-morpholinium iminium.H2O",
                "note": "Dehydration gives cyclic iminium ion."
            },
            {
                "step_index": 3,
                "reaction_example": "iminium.NaBH(OAc)3>>N-cyclohexyl-morpholinium ammonium",
                "note": "Mild hydride (NaBH(OAc)3) reduces iminium C."
            },
            {
                "step_index": 4,
                "reaction_example": "ammonium.base>>N-cyclohexylmorpholine",
                "note": "Deprotonation gives neutral tertiary amine product."
            }
        ]
    },
    "rt_080": {
        "example_notes": "4-hydroxybutanoic acid + H+ gives gamma-butyrolactone by intramolecular lactonization.",
        "current_state_example": ["HO-(CH2)3-CO2H (4-hydroxybutanoic acid, gamma-hydroxy acid)", "H+ (catalyst)"],
        "resulting_state_example": ["gamma-butyrolactone (GBL, 4-membered ring lactone)", "H2O"],
        "example_mechanism_steps": [
            {
                "step_index": 1,
                "reaction_example": "HO(CH2)3CO2H.H+>>activated carboxylic acid (protonated carbonyl)",
                "note": "Acid protonates carbonyl of carboxylic acid; electrophilic toward internal OH."
            },
            {
                "step_index": 2,
                "reaction_example": "activated hydroxy acid>>gamma-butyrolactone.H2O",
                "note": "Internal OH attacks activated carboxyl C; ring closes; water expelled; GBL."
            }
        ]
    }
}

def build_examples_json():
    with open('/Users/scottreed/PycharmProjects/professor-wiggum/training_data/reaction_type_templates_rewrite.json') as f:
        source = json.load(f)

    templates = source['templates']
    out_templates = []

    for t in templates:
        tid = t['type_id']
        ex = EXAMPLES.get(tid)
        if ex is None:
            print(f"WARNING: No example for {tid}")
            continue

        steps = ex['example_mechanism_steps']
        generic_steps = []
        for s in steps:
            generic_steps.append({
                "step_index": s['step_index'],
                "reaction_generic": s['reaction_example'],
                "note": s['note']
            })

        entry = {
            "type_id": tid,
            "label_exact": t['label_exact'],
            "slug": t['slug'],
            "canonical_group": t['canonical_group'],
            "example_notes": ex['example_notes'],
            # Example fields (canonical names)
            "current_state_example": ex['current_state_example'],
            "resulting_state_example": ex['resulting_state_example'],
            "example_mechanism_steps": steps,
            # Aliased fields so existing render script works unchanged
            "current_state_generic": ex['current_state_example'],
            "resulting_state_generic": ex['resulting_state_example'],
            "generic_mechanism_steps": generic_steps,
        }
        out_templates.append(entry)

    out = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "count": len(out_templates),
            "description": "Concrete examples for each reaction type template. R-group notation replaced with specific simple molecules. Includes both _example and _generic aliased fields for render script compatibility.",
            "r_group_rules": {
                "R": "CH3 (methyl) unless context suggests otherwise",
                "R_prime": "C2H5 (ethyl)",
                "Ar": "Ph (phenyl)",
                "HetAr": "2-pyridyl",
                "X_halide": "Cl or Br depending on context",
                "Base": "NaOH, K2CO3, KOtBu, LDA depending on context",
                "allyl": "CH2CH=CH2",
                "epoxide": "ethylene oxide or propylene oxide",
                "stabilized_enolate": "acetylacetonate anion"
            }
        },
        "templates": out_templates
    }

    out_path = '/Users/scottreed/PycharmProjects/professor-wiggum/training_data/reaction_type_templates_examples.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"Written {len(out_templates)} templates to {out_path}")
    return len(out_templates)

if __name__ == '__main__':
    n = build_examples_json()
    print(f"Done: {n} templates")
