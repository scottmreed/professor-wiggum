# Prepare Inputs Skill

Convert non-SMILES chemical input to canonical SMILES and prepare test case templates.

## Supported Input Formats

- **SMILES** -- validated and canonicalised by RDKit
- **InChI** -- converted via RDKit `MolFromInchi`
- **MOL / SDF block** -- converted via RDKit `MolFromMolBlock`
- **Common name** -- looked up via PubChem PUG REST API (requires network)

## API

### Convert Inputs

`POST /api/convert_inputs`

```json
{
  "starting_materials": ["ethanol", "InChI=1S/C2H4O2/c1-2(3)4/h1H3,(H,3,4)"],
  "products": ["CC(=O)OCC"]
}
```

Returns canonical SMILES for each input plus any conversion errors.

### Import Template

`POST /api/eval_sets/import_template`

Upload a completed `templates/test_case_template.json` (or YAML via the
`yaml_text` field) to create a named eval set importable into the existing
eval pipeline.

## Templates

See `templates/test_case_template.json` and `templates/test_case_template.yaml`
for fill-in templates that include model selection and optional known-answer
verification steps.

## Notes

- PubChem lookups are optional. If the network is unavailable, the endpoint
  returns an error for that compound only; other compounds are still converted.
- All output SMILES are validated by RDKit before being returned.
- Verified mechanism steps in the template are stored in `expected_json` in
  `eval_set_cases` and compared during eval run grading.
