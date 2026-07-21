import os
import tempfile
import json
import csv
from scripts.generate_inputs import clean_filename, generate_json_inputs
from scripts.rescore_gnina import parse_mmcif_atoms, parse_gnina_output, write_pdb
from scripts.rank_candidates import parse_af3_summary, generate_svg_slopegraph, load_true_positives

# Test clean_filename
def test_clean_filename():
    assert clean_filename("TF_Name!") == "TF_Name_"
    assert clean_filename("TF-Name123") == "TF-Name123"
    assert clean_filename("TF/Name*") == "TF_Name_"

# Test input generator
def test_generate_json_inputs():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "pairings.csv")
        out_dir = os.path.join(tmpdir, "jsons")
        
        # Create mock pairings.csv
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['TF_Name', 'TF_Sequence', 'Ligand_Name', 'Ligand_SMILES'])
            writer.writerow(['TF1', 'MVKV', 'LigA', 'CC'])
            writer.writerow(['TF2', 'MVKV', 'LigB', 'CCC'])
            
        success = generate_json_inputs(csv_path, out_dir)
        assert success is True
        
        # Verify JSON outputs
        json1_path = os.path.join(out_dir, "TF1_LigA.json")
        assert os.path.exists(json1_path)
        with open(json1_path, 'r') as f:
            data = json.load(f)
            assert data['dialect'] == 'alphafold3'
            assert data['sequences'][0]['protein']['sequence'] == 'MVKV'
            assert data['sequences'][1]['ligand']['smiles'] == 'CC'

# Test parse_mmcif_atoms
def test_parse_mmcif_atoms():
    mock_cif = """
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.auth_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
ATOM 1 N N MET A 1 1.000 2.000 3.000 1.00 20.00
HETATM 2 C C1 LIG B 2 10.000 11.000 12.000 1.00 15.00
"""
    atoms = parse_mmcif_atoms(mock_cif)
    assert len(atoms) == 2
    assert atoms[0]['_atom_site.label_comp_id'] == 'MET'
    assert atoms[0]['_atom_site.label_asym_id'] == 'A'
    assert atoms[1]['_atom_site.label_comp_id'] == 'LIG'
    assert atoms[1]['_atom_site.label_asym_id'] == 'B'

# Test write_pdb
def test_write_pdb():
    mock_atoms = [
        {
            '_atom_site.group_PDB': 'ATOM',
            '_atom_site.id': '1',
            '_atom_site.label_atom_id': 'N',
            '_atom_site.label_comp_id': 'MET',
            '_atom_site.label_asym_id': 'A',
            '_atom_site.auth_seq_id': '1',
            '_atom_site.Cartn_x': '1.000',
            '_atom_site.Cartn_y': '2.000',
            '_atom_site.Cartn_z': '3.000',
            '_atom_site.occupancy': '1.00',
            '_atom_site.B_iso_or_equiv': '20.00',
            '_atom_site.type_symbol': 'N'
        },
        {
            '_atom_site.group_PDB': 'HETATM',
            '_atom_site.id': '2',
            '_atom_site.label_atom_id': 'C',
            '_atom_site.label_comp_id': 'LIG',
            '_atom_site.label_asym_id': 'B',
            '_atom_site.auth_seq_id': '2',
            '_atom_site.Cartn_x': '10.000',
            '_atom_site.Cartn_y': '11.000',
            '_atom_site.Cartn_z': '12.000',
            '_atom_site.occupancy': '1.00',
            '_atom_site.B_iso_or_equiv': '15.00',
            '_atom_site.type_symbol': 'C'
        }
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        prot_path = os.path.join(tmpdir, "protein.pdb")
        lig_path = os.path.join(tmpdir, "ligand.pdb")
        
        write_pdb(mock_atoms, "A", prot_path)
        write_pdb(mock_atoms, "B", lig_path)
        
        assert os.path.exists(prot_path)
        assert os.path.exists(lig_path)
        
        with open(prot_path, 'r') as f:
            lines = f.readlines()
            assert any("MET A   1" in line for line in lines)
            assert not any("LIG B" in line for line in lines)

# Test parse_gnina_output
def test_parse_gnina_output():
    # Test flat format
    stdout_flat = """
Affinity: -8.123
CNNscore: 0.9234
CNNaffinity: 7.234
"""
    score, affinity = parse_gnina_output(stdout_flat)
    assert score == 0.9234
    assert affinity == 7.234
    
    # Test table format
    stdout_table = """
mode |   affinity | intramolecular | cnn_pose_score | cnn_affinity
-----+------------+----------------+----------------+-------------
   1 |     -8.123 |          0.000 |         0.9234 |        7.234
   2 |     -7.456 |          0.000 |         0.7812 |        6.123
"""
    score_t, affinity_t = parse_gnina_output(stdout_table)
    assert score_t == 0.9234
    assert affinity_t == 7.234

# Test SVG generation
def test_generate_svg_slopegraph():
    mock_val_data = [
        {"TF_Ligand": "TF1_LigA", "af3_rank": 3, "consensus_rank": 1, "total_candidates": 5},
        {"TF_Ligand": "TF2_LigB", "af3_rank": 1, "consensus_rank": 1, "total_candidates": 5}
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        svg_path = os.path.join(tmpdir, "plot.svg")
        generate_svg_slopegraph(mock_val_data, svg_path)
        assert os.path.exists(svg_path)
        with open(svg_path, 'r') as f:
            content = f.read()
            assert "<svg" in content
            assert "True Positive Rank" in content
            assert "TF1_LigA" in content
