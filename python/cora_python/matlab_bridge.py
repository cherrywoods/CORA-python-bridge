from pathlib import Path


# CORA root directory (parent of the python/ directory)
CORA_ROOT = str(Path(__file__).resolve().parent.parent.parent)


class MatlabBridge:
    """Manages a MATLAB engine session with CORA on the path."""

    def __init__(self):
        self._engine = None

    @property
    def engine(self):
        if self._engine is None:
            raise RuntimeError("MATLAB engine not started. Call start() first.")
        return self._engine

    def start(self, matlab_path: str | None = None):
        """Start the MATLAB engine and add CORA to the path."""
        import matlab.engine

        if matlab_path:
            self._engine = matlab.engine.start_matlab(
                f"-nodesktop -nosplash -r \"addpath('{matlab_path}')\""
            )
        else:
            self._engine = matlab.engine.start_matlab("-nodesktop -nosplash")

        # Add CORA to MATLAB path
        self._engine.addpath(self._engine.genpath(CORA_ROOT), nargout=0)

        # Add the _matlab helper directory
        matlab_helpers = str(Path(__file__).resolve().parent / "_matlab")
        self._engine.addpath(matlab_helpers, nargout=0)

    def stop(self):
        """Stop the MATLAB engine."""
        if self._engine is not None:
            self._engine.quit()
            self._engine = None

    def add_to_path(self, directory: str):
        """Add a directory to the MATLAB path."""
        self.engine.addpath(directory, nargout=0)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
