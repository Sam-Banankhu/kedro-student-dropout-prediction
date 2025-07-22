from kedro.framework.hooks import hook_impl
from kedro_viz import server
from pathlib import Path
import logging
import webbrowser
import time

logger = logging.getLogger(__name__)

class VisualizationHooks:
    """Custom hooks for visualization and monitoring."""
    
    def __init__(self):
        self._viz_process = None

    @hook_impl
    def after_pipeline_run(self, run_params: dict, pipeline, catalog):
        """Launch Kedro-Viz after prediction pipeline completes."""
        if run_params.get("pipeline_name") == "prediction":
            try:
                logger.info("Preparing pipeline visualization...")
                viz_path = Path("data/08_reporting/pipeline_visualization.html")
                
                # Start the server without auto-opening browser
                self._viz_process = server.run_server(
                    save_file=viz_path,
                    port=4141,
                    browser=False
                )
                
                # Give server time to start
                time.sleep(2)
                webbrowser.open("http://127.0.0.1:4141")
                
            except Exception as e:
                logger.error(f"Visualization failed: {str(e)}")

    @hook_impl
    def after_context_teardown(self):
        """Clean up the visualization server."""
        if self._viz_process:
            self._viz_process.terminate()

    @hook_impl
    def on_node_error(self, error):
        """Simplified error handling."""
        logger.error(f"Node failed with error: {str(error)}")