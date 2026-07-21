import os
import subprocess
import pytest
from unittest.mock import patch, MagicMock

from scripts.run_pipeline_euler import run_pipeline

def test_run_pipeline_dry_run(tmp_path):
    jsons_dir = tmp_path / "alphafold3_jsons"
    datafill_dir = tmp_path / "alphafold3_datafill"
    predictions_dir = tmp_path / "alphafold3_predictions"
    processed_dir = tmp_path / "data" / "processed"
    
    jsons_dir.mkdir(parents=True)
    (jsons_dir / "test_pair.json").write_text('{"name": "test_pair"}')
    
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="Success")
        
        # Test dry-run mode
        success = run_pipeline(
            input_jsons_dir=str(jsons_dir),
            datafill_dir=str(datafill_dir),
            predictions_dir=str(predictions_dir),
            config_path="config/config.yaml",
            dry_run=True
        )
        assert success is True
        assert mock_run.call_count == 0  # Dry run shouldn't execute shell commands
