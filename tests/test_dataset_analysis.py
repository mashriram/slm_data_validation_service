import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.services.dataset_analysis import dataset_analysis_service

@pytest.mark.asyncio
async def test_analyze_dataset_success():
    # Mock get_dataset_config_names
    with patch("app.services.dataset_analysis.get_dataset_config_names", return_value=["default"]):
        # Mock load_dataset_builder
        with patch("app.services.dataset_analysis.load_dataset_builder") as mock_builder:
            mock_info = MagicMock()
            mock_info.description = "Test Description"
            mock_info.citation = "Test Citation"
            mock_info.features = {"text": "string"}
            mock_info.splits = {"train": MagicMock(num_examples=100)}
            mock_info.download_size = 1000
            mock_info.dataset_size = 2000
            mock_builder.return_value.info = mock_info
            
            # Mock load_dataset (streaming)
            with patch("app.services.dataset_analysis.load_dataset") as mock_load_dataset:
                mock_stream = MagicMock()
                # diverse texts
                mock_stream.take.return_value = [{"text": "hello world"}, {"text": "foo bar"}, {"text": "different thing"}]
                mock_load_dataset.return_value = mock_stream
                
                # Mock SentenceTransformer
                with patch("app.services.dataset_analysis.SentenceTransformer") as mock_model_cls:
                    mock_model = MagicMock()
                    # Mock embeddings (3 samples, dim=2)
                    mock_model.encode.return_value = [[0.1, 0.9], [0.9, 0.1], [0.5, 0.5]] 
                    dataset_analysis_service.embedding_model = mock_model
                    
                    dataset_configs = [{"id": "fake/dataset", "config": "default", "split": "train"}]
                    result = await dataset_analysis_service.analyze_datasets(dataset_configs, target_model="meta-llama/Llama-2-7b-hf")
                    
                    assert "analyzed_datasets" in result, f"Result contained error: {result.get('error')}"
                    assert result["analyzed_datasets"][0]["dataset_id"] == "fake/dataset"
                    assert "text" in result["columns"]
                    assert "semantic_metrics" in result
                    assert "recommendations" in result
                    assert "rating_score" in result
                    assert "perplexity_score" in result

@pytest.mark.asyncio
async def test_analyze_dataset_empty():
    with patch("app.services.dataset_analysis.get_dataset_config_names", return_value=["default"]):
        with patch("app.services.dataset_analysis.load_dataset_builder") as mock_builder:
            mock_info = MagicMock()
            mock_info.splits = {"train": MagicMock(num_examples=100)}
            mock_builder.return_value.info = mock_info
            
            with patch("app.services.dataset_analysis.load_dataset") as mock_load_dataset:
                mock_stream = MagicMock()
                mock_stream.take.return_value = [] # Empty
                mock_load_dataset.return_value = mock_stream

                dataset_configs = [{"id": "empty/dataset", "config": "default", "split": "train"}]
                result = await dataset_analysis_service.analyze_datasets(dataset_configs, target_model="fake-model")
                
                assert "error" in result
                assert "appear to be empty" in result["error"]
