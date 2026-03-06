#!/usr/bin/env python3
"""
Script to visualize mechanistic steps and electron pushes from mechanism_examples.json using RDKit.
Generates PNG files for each step_index in each reaction.

Usage:
    python data/visualize_reactions.py

Requirements:
    - RDKit (install via: conda install -c conda-forge rdkit)
    - mechanism_examples.json in the same directory

Output:
    - One PNG file per mechanistic step, named as: {reaction_id}_step_{step_index}.png
    - Images show current_state → resulting_state with reaction arrow
    - Includes electron push information in filename and console output
    - Files are saved in the data/ folder

Note:
    Generated PNG files are gitignored as they can be recreated.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any

try:
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem
except ImportError:
    print("RDKit not found. Please install RDKit:")
    print("conda install -c conda-forge rdkit")
    exit(1)


def load_reactions() -> List[Dict[str, Any]]:
    """Load reactions from mechanism_examples.json."""
    json_path = Path(__file__).parent / "mechanism_examples.json"
    with open(json_path, 'r') as f:
        return json.load(f)


def smiles_to_mol(smiles: str) -> Chem.Mol:
    """Convert SMILES to RDKit molecule, handling errors gracefully."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Warning: Could not parse SMILES: {smiles}")
            return None
        # Add hydrogens for better visualization
        mol = Chem.AddHs(mol)
        # Generate 2D coordinates
        AllChem.Compute2DCoords(mol)
        return mol
    except Exception as e:
        print(f"Error processing SMILES {smiles}: {e}")
        return None


def format_electron_pushes(electron_pushes: List[Dict[str, Any]]) -> str:
    """Format electron push information for display."""
    if not electron_pushes:
        return "No electron pushes defined"

    pushes = []
    for push in electron_pushes:
        start = push.get('start_atom', '?')
        end = push.get('end_atom', '?')
        electrons = push.get('electrons', '?')
        desc = push.get('description', 'electron movement')
        pushes.append(f"Atom {start} → {end} ({electrons}e: {desc})")

    return "; ".join(pushes)


def create_step_image(current_state: List[str], resulting_state: List[str],
                     reaction_id: str, step_index: int, name: str,
                     electron_pushes: List[Dict[str, Any]]) -> str:
    """
    Create a mechanistic step visualization image.

    Args:
        current_state: List of SMILES strings for molecules in current state
        resulting_state: List of SMILES strings for molecules after step
        reaction_id: Unique identifier for the reaction
        step_index: Index of this mechanistic step
        name: Human-readable name for the reaction
        electron_pushes: List of electron push dictionaries

    Returns:
        Filename of the saved image
    """
    # Convert SMILES to molecules
    reactant_mols = []
    product_mols = []

    for smiles in current_state:
        mol = smiles_to_mol(smiles)
        if mol:
            reactant_mols.append(mol)

    for smiles in resulting_state:
        mol = smiles_to_mol(smiles)
        if mol:
            product_mols.append(mol)

    if not reactant_mols or not product_mols:
        print(f"Skipping {reaction_id} step {step_index}: no valid molecules found")
        return None

    # Create reaction object for this step
    rxn = AllChem.ChemicalReaction()

    # Add reactants (current state) and products (resulting state)
    for mol in reactant_mols:
        rxn.AddReactantTemplate(mol)
    for mol in product_mols:
        rxn.AddProductTemplate(mol)

    # Generate reaction depiction
    try:
        # Draw the reaction
        img = Draw.ReactionToImage(rxn)

        # Save the image
        output_dir = Path(__file__).parent
        filename = f"{reaction_id}_step_{step_index}.png"
        output_path = output_dir / filename

        img.save(output_path)

        # Display electron push information
        push_info = format_electron_pushes(electron_pushes)
        print(f"Saved: {filename} - {name}")
        print(f"  Electron pushes: {push_info}")

        return filename

    except Exception as e:
        print(f"Error creating image for {reaction_id} step {step_index}: {e}")
        return None


def main():
    """Main function to process all mechanistic steps."""
    print("Loading reactions from mechanism_examples.json...")

    try:
        reactions = load_reactions()
        print(f"Found {len(reactions)} reactions")

        output_dir = Path(__file__).parent
        print(f"Saving step images to: {output_dir.absolute()}")

        successful = 0
        failed = 0
        total_steps = 0

        for reaction in reactions:
            reaction_id = reaction.get('id', 'unknown')
            name = reaction.get('name', 'Unknown reaction')

            # Get verified mechanism steps
            verified_mechanism = reaction.get('verified_mechanism', {})
            steps = verified_mechanism.get('steps', [])

            if not steps:
                print(f"Skipping {reaction_id}: no verified mechanism steps")
                failed += 1
                continue

            print(f"\nProcessing {reaction_id}: {name}")
            print(f"Found {len(steps)} mechanistic step(s)")

            for step in steps:
                step_index = step.get('step_index', 0)
                current_state = step.get('current_state', [])
                resulting_state = step.get('resulting_state', [])
                electron_pushes = step.get('electron_pushes', [])

                total_steps += 1

                filename = create_step_image(
                    current_state, resulting_state,
                    reaction_id, step_index, name,
                    electron_pushes
                )

                if filename:
                    successful += 1
                else:
                    failed += 1

        print(f"\nProcessing complete:")
        print(f"  Total steps processed: {total_steps}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")

    except FileNotFoundError:
        print("Error: mechanism_examples.json not found in the same directory")
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()