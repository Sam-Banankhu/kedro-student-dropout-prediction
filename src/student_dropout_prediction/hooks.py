from kedro.framework.hooks import hook_impl
from kedro_viz import server
from pathlib import Path
import webbrowser
import time
import logging

logger = logging.getLogger(__name__)

class VisualizationHooks:
    """Custom hooks for prediction pipeline visualization."""
    
    def __init__(self):
        self.viz_process = None
    
    @hook_impl
    def after_pipeline_run(self, run_params: dict):
        """Launch Kedro-Viz after prediction pipeline runs."""
        if run_params.get("pipeline_name") == "prediction":
            # Save visualization to HTML
            viz_path = Path("data/08_reporting/predictions_viz.html")
            logger.info("Generating pipeline visualization...")
            
            # Launch Kedro-Viz server
            self.viz_process = server.run_server(
                port=4141,
                save_file=viz_path,
                browser=False  # We'll open manually
            )
            
            # Open in browser after short delay
            time.sleep(2)
            webbrowser.open("http://127.0.0.1:4141")
            
    @hook_impl
    def after_context_teardown(self):
        """Clean up the Viz server when done."""
        if self.viz_process:
            self.viz_process.terminate()
            
            
    @hook_impl
    def before_pipeline_run(self, run_params: dict):
        """Log pipeline start."""
        logger.info(f"Starting pipeline: {run_params['pipeline_name']}")
        
    @hook_impl 
    def on_node_error(self, error: Exception, node_name: str):
        """Send alerts on failures."""
        logger.error(f"Node failed: {node_name} - {str(error)}")
        # Add email/Teams alert here if needed

    @hook_impl
    def after_catalog_created(self, catalog):
        """Validate catalog entries."""
        if "new_student_data" not in catalog:
            logger.warning("Missing new_student_data in catalog!")