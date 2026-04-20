import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
import pandas as pd
import json
import logging
import httpx
from datasets import get_dataset_config_names, get_dataset_split_names

logger = logging.getLogger(__name__)

def fetch_dataset_configs(dataset_id):
    """Fetches available configs and splits for a given dataset ID."""
    if not dataset_id:
        return gr.update(choices=[], value=None), gr.update(choices=[], value=None)
    
    try:
        configs = get_dataset_config_names(dataset_id)
        if not configs:
            configs = ["default"]
            
        splits = get_dataset_split_names(dataset_id, configs[0])
        if not splits:
             splits = ["train"]
             
        return gr.update(choices=configs, value=configs[0], interactive=True), gr.update(choices=splits, value=splits[0], interactive=True)
    except Exception as e:
        logger.warning(f"Could not fetch details for {dataset_id}: {e}")
        return gr.update(choices=["default"], value="default", interactive=True), gr.update(choices=["train", "test", "validation"], value="train", interactive=True)

def fetch_dataset_splits(dataset_id, config_name):
    if not dataset_id or not config_name:
        return gr.update(choices=[], value=None)
    try:
        splits = get_dataset_split_names(dataset_id, config_name)
        if not splits:
            splits = ["train"]
        return gr.update(choices=splits, value=splits[0], interactive=True)
    except Exception as e:
        logger.warning(f"Could not fetch splits for {dataset_id}, {config_name}: {e}")
        return gr.update(choices=["train", "test", "validation"], value="train", interactive=True)

def add_dataset_to_list(current_list, d_id, config, split):
    if not d_id:
        return current_list
    
    new_entry = {"id": d_id, "config": config if config else "default", "split": split if split else "train"}
    
    # Parse current list if it's a string from the UI state
    if isinstance(current_list, str):
        try:
            parsed = json.loads(current_list) if current_list else []
        except:
             parsed = []
    else:
        parsed = current_list or []
        
    parsed.append(new_entry)
    
    # Generate display text
    display_text = "\n".join([f"- {d['id']} (Config: {d['config']}, Split: {d['split']})" for d in parsed])
    return json.dumps(parsed), display_text

def clear_dataset_list():
     return "[]", "No datasets added yet."

async def analyze_datasets_gradio(dataset_state_json, target_model, request: gr.Request):
    try:
        dataset_configs = json.loads(dataset_state_json) if dataset_state_json else []
    except:
        dataset_configs = []

    if not dataset_configs:
        return "⚠️ Please add at least one dataset before analyzing.", None, None, None, gr.update(value=0)

    if not target_model:
        target_model = "meta-llama/Meta-Llama-3-8B"

    # Use actual host url if passed from request
    api_url = "http://127.0.0.1:8000/api/v1/dataset/analyze"
    if request and request.headers and 'host' in request.headers:
        # If running via FastAPI mount, route back internally via network proxy standard
        # But localhost is safest for local backend dev. Let's just use localhost.
         pass

    try:
        # Call backend via API
        async with httpx.AsyncClient(timeout=300.0) as client:
             response = await client.post(
                 api_url, 
                 json={"datasets": dataset_configs, "target_model": target_model}
             )
             
             if response.status_code != 200:
                  error_detail = response.json().get("detail", "Unknown API error")
                  return f"❌ API Error: {error_detail}", None, None, None, gr.update(value=0)
                  
             result = response.json()
             
        # 1. Format Summary
        summary_md = f"### 📄 Dataset Overview\n"
        for idx, d in enumerate(result.get('analyzed_datasets', [])):
             summary_md += f"**{idx+1}.** `{d['dataset_id']}` (Config: `{d['config']}`, Split: `{d['split']}`) - Rows sample: {d['rows']:,}\n"
             
        summary_md += f"\n**Total Analyzed Rows:** {result.get('total_rows', 0):,}\n"
        summary_md += f"**Target Model:** `{result.get('target_model', 'N/A')}`\n"
        summary_md += f"**Analyzed Columns:** `{', '.join(result.get('columns', []))}`\n"
        
        # 2. Format Recommendations & Rating deductions
        rec_md = "### 💡 Analysis & Deductions\n\n"
        
        recs = result.get('recommendations', [])
        for rec in recs:
            rec_md += f"- ✅ {rec}\n"
            
        deductions = result.get('rating_deductions', [])
        for d in deductions:
             rec_md += f"- ⚠️ {d}\n"
             
        missing = result.get('missing_characteristics', [])
        if missing:
             rec_md += "\n**🚨 What this dataset combo is MISSING:**\n"
             for m in missing:
                  rec_md += f"- {m}\n"
        
        if not recs and not deductions and not missing:
            rec_md += "*No specific insights generated.*"
            

        # 3. Format Metrics Table
        metrics_data = {
             "Metric": ["Semantic Diversity (0-1)", "Outlier Ratio", "Simulated Perplexity", "Avg Sequence Length", "Max Sequence Length"],
             "Value": [
                 round(result.get('semantic_metrics', {}).get('diversity_score', 0), 3),
                 round(result.get('semantic_metrics', {}).get('outlier_ratio', 0), 3),
                 round(result.get('perplexity_score', 0), 2),
                 round(result.get('sequence_metrics', {}).get('avg_length', 0), 1),
                 result.get('sequence_metrics', {}).get('max_length', 0)
             ]
        }
        metrics_df = pd.DataFrame(metrics_data)
        
        # 4. JSON
        json_output = json.dumps(result, indent=2)
        
        # 5. Rating Number
        rating = int(result.get('rating_score', 0))
        
        # We return the Number component update based on rating
        return summary_md, rec_md, metrics_df, json_output, rating

    except Exception as e:
        logger.exception("Error in UI processing")
        return f"🚨 Unexpected Error: {str(e)}", None, None, None, gr.update(value=0)

custom_css = """
.gradio-container { font-family: 'Inter', sans-serif; }
.main-title { text-align: center; margin-bottom: 0.2rem; font-weight: 800; font-size: 2.5rem;}
.sub-title { text-align: center; margin-bottom: 2rem; font-size: 1.2rem; opacity: 0.8; }
.panel { border: 1px solid var(--border-color-primary); border-radius: 12px; padding: 1.5rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); }
.score-box { text-align: center; }
.score-value { font-size: 4rem; font-weight: bold; color: #3b82f6; }
"""

def create_gradio_interface():
    with gr.Blocks(css=custom_css, title="SLM Data 360🚀") as demo:
        gr.Markdown("<h1 class='main-title'>SLM Dataset 360° Studio</h1>")
        gr.Markdown("<p class='sub-title'>Interleave, validate, and rate Hugging Face dataset combinations against target architectures.</p>")
        
        # Hidden state to store exactly what to send to the backend
        dataset_state = gr.State("[]")
        
        with gr.Row():
            with gr.Column(scale=1, elem_classes="panel"):
                gr.Markdown("### 🛠️ Configuration")
                
                with gr.Group():
                    gr.Markdown("#### Add Datasets to Interleave")
                    dataset_id_input = gr.Textbox(
                        label="Hugging Face Dataset ID", 
                        placeholder="e.g., imdb or HuggingFaceH4/ultrachat_200k",
                    )
                    
                    with gr.Row():
                        config_input = gr.Dropdown(label="Subset (Config)", choices=["default"], value="default", allow_custom_value=True)
                        split_input = gr.Dropdown(label="Split", choices=["train", "test", "validation"], value="train", allow_custom_value=True)
                    
                    dataset_id_input.blur(
                        fn=fetch_dataset_configs,
                        inputs=[dataset_id_input],
                        outputs=[config_input, split_input]
                    )
                    
                    config_input.change(
                        fn=fetch_dataset_splits,
                        inputs=[dataset_id_input, config_input],
                        outputs=[split_input]
                    )
                    
                    add_btn = gr.Button("➕ Add to Analysis List", size="sm")
                    
                gr.Markdown("#### Select Target Model")
                target_model_input = gr.Dropdown(
                    choices=[
                        "meta-llama/Meta-Llama-3-8B",
                        "mistralai/Mistral-7B-v0.1",
                        "google/gemma-7b",
                        "microsoft/phi-2"
                    ],
                    value="meta-llama/Meta-Llama-3-8B",
                    allow_custom_value=True,
                    label="Target SLM Architecture",
                    info="Used for tokenizer sequence analysis & perplexity."
                )
                
                gr.Markdown("#### Queue")
                dataset_list_display = gr.Markdown("No datasets added yet.")
                clear_btn = gr.Button("🗑️ Clear List", size="sm")
                
                gr.Markdown("---")
                analyze_btn = gr.Button("🚀 Run 360° Analysis", variant="primary", size="lg")
                
                gr.Markdown("---")
                gr.Markdown("### 🎯 360° Compatibility Score")
                rating_output = gr.Number(label="Score (0-100)", value=0, interactive=False, elem_classes="score-value")
            
            with gr.Column(scale=2):
                with gr.Row():
                    summary_output = gr.Markdown("### 📄 Setup\n*Waiting for input...*", elem_classes="panel")
                with gr.Row():
                    recommendations_output = gr.Markdown("### 💡 Insights & Deductions\n*Waiting for input...*", elem_classes="panel")
        
        with gr.Row():
            with gr.Column(elem_classes="panel"):
                gr.Markdown("### 📈 Advanced Metrics (Perplexity, Semantics, Lengths)")
                metrics_output = gr.Dataframe(
                    label="Metrics", 
                    interactive=False
                )
            
        with gr.Row():
             with gr.Accordion("Raw Analysis JSON (Advanced)", open=False):
                 json_output = gr.Code(label="JSON Result", language="json")

        # Event wiring
        add_btn.click(
            fn=add_dataset_to_list,
            inputs=[dataset_state, dataset_id_input, config_input, split_input],
            outputs=[dataset_state, dataset_list_display]
        )
        
        clear_btn.click(
            fn=clear_dataset_list,
            inputs=[],
            outputs=[dataset_state, dataset_list_display]
        )

        analyze_btn.click(
            fn=analyze_datasets_gradio,
            inputs=[dataset_state, target_model_input],
            outputs=[summary_output, recommendations_output, metrics_output, json_output, rating_output]
        )
        
    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860)
