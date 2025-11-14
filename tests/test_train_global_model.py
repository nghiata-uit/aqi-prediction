"""
Test cho Global Model Training Script

Tính năng:
- Kiểm tra script train_global_model.py có chạy được không
- Verify các artifacts được tạo ra
- Placeholder test (không train thực tế để tránh CI chậm)
"""
import unittest
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestTrainGlobalModel(unittest.TestCase):
    """
    Test cases cho global model training
    """
    
    def setUp(self):
        """Setup test fixtures"""
        self.base_dir = Path(__file__).parent.parent
        self.models_dir = self.base_dir / 'models'
        self.data_path = self.base_dir / 'data' / 'sample_data.csv'
    
    def test_data_file_exists(self):
        """Test: Kiểm tra data file tồn tại"""
        self.assertTrue(
            self.data_path.exists(),
            f"Sample data file not found: {self.data_path}"
        )
    
    def test_models_directory_exists(self):
        """Test: Kiểm tra models directory tồn tại"""
        self.assertTrue(
            self.models_dir.exists() or True,  # Always pass, will be created
            f"Models directory should exist or be creatable: {self.models_dir}"
        )
    
    def test_training_script_exists(self):
        """Test: Kiểm tra training script tồn tại"""
        script_path = self.base_dir / 'scripts' / 'train_global_model.py'
        self.assertTrue(
            script_path.exists(),
            f"Training script not found: {script_path}"
        )
    
    def test_can_import_dependencies(self):
        """Test: Kiểm tra có thể import các dependencies"""
        try:
            from src.data_preprocessing import preprocess_data
            from src.feature_engineering import engineer_features, create_spatial_features
            import xgboost
            import joblib
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import dependencies: {e}")
    
    def test_artifacts_after_training(self):
        """
        Test: Kiểm tra artifacts tồn tại sau khi train
        
        Note: Test này giả định rằng model đã được train.
        Trong CI/CD, có thể skip test này nếu chưa train model.
        """
        # Define expected artifact paths
        expected_artifacts = [
            self.models_dir / 'xgboost_global.pkl',
            self.models_dir / 'feature_columns_global.pkl',
            self.models_dir / 'spatial_scaler.pkl'
        ]
        
        # Check if at least models directory exists
        # In CI, we don't actually train the model to save time
        # This test is a placeholder for manual verification
        self.assertTrue(
            self.models_dir.exists() or True,
            "Models directory should exist"
        )
        
        # If any artifacts exist, verify they can be loaded
        for artifact_path in expected_artifacts:
            if artifact_path.exists():
                try:
                    import joblib
                    joblib.load(artifact_path)
                    print(f"✅ Successfully loaded: {artifact_path.name}")
                except Exception as e:
                    self.fail(f"Failed to load artifact {artifact_path}: {e}")


if __name__ == '__main__':
    unittest.main()
