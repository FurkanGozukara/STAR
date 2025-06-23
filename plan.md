Of course. Here is a detailed, step-by-step plan to implement a robust, global cancellation feature for your application.

---
# PLAN.md: Implementing a Global Cancellation Feature

## **🎯 Objective**

To introduce a single, reliable "Cancel" button in the Gradio UI that can gracefully terminate any long-running process (single upscale, batch processing, RIFE, face restoration, etc.) and return the application to a ready state.

## **⚙️ Core Strategy**

We will implement a central cancellation manager that uses a `threading.Event` to signal a cancellation request. All long-running loops and processes will periodically check this signal. If a cancellation is requested, they will raise a custom `CancelledError` exception, which will be caught at the top level to ensure a clean exit and UI reset.

---

## **Phase 1: Foundation - The Cancellation Framework**

This phase lays the groundwork for the cancellation mechanism without integrating it into the processing loops yet.

### **Step 1.1: Create the Cancellation Manager**

1.  Create a new file: `logic/cancellation_manager.py`.
2.  Add the following code to it. This class will manage the cancellation state and hold references to any running subprocesses.

    ```python
    # logic/cancellation_manager.py
    import threading
    import subprocess
    from typing import Optional

    class CancelledError(Exception):
        """Custom exception to signal a graceful cancellation."""
        pass

    class CancellationManager:
        def __init__(self):
            self._cancel_event = threading.Event()
            self._active_process: Optional[subprocess.Popen] = None
            self._lock = threading.Lock()

        def request_cancellation(self):
            """Signal that a cancellation has been requested."""
            self._cancel_event.set()
            with self._lock:
                if self._active_process:
                    try:
                        self._active_process.terminate()
                    except Exception:
                        pass

        def is_cancelled(self) -> bool:
            """Check if cancellation has been requested."""
            return self._cancel_event.is_set()

        def check_cancel(self):
            """Check for cancellation and raise CancelledError if requested."""
            if self.is_cancelled():
                raise CancelledError("Processing was cancelled by the user.")

        def reset(self):
            """Reset the cancellation state for a new operation."""
            self._cancel_event.clear()
            with self._lock:
                self._active_process = None

        def set_active_process(self, process: subprocess.Popen):
            """Store the currently running subprocess."""
            with self._lock:
                self._active_process = process

        def clear_active_process(self):
            """Clear the stored subprocess."""
            with self._lock:
                self._active_process = None

    # Global instance to be used across the application
    cancellation_manager = CancellationManager()
    ```

### **Step 1.2: Add the Cancel Button to the UI**

1.  In `secourses_app.py`, locate the main "Upscale Video" button.
2.  Add a new "Cancel" button next to it, initially hidden.

    ```python
    # In secourses_app.py, inside the main tab layout
    with gr.Row():
        upscale_button = gr.Button("Upscale Video", variant="primary", icon="icons/upscale.png")
        cancel_button = gr.Button("Cancel", variant="stop", visible=False) # New button
    ```

### **Step 1.3: Implement the Cancellation Logic**

1.  In `secourses_app.py`, import the new manager:
    ```python
    from logic.cancellation_manager import cancellation_manager, CancelledError
    ```
2.  Create a click handler for the new "Cancel" button.
    ```python
    def cancel_processing():
        logger.warning("CANCEL button clicked. Requesting cancellation.")
        cancellation_manager.request_cancellation()
        return "Cancellation requested..."

    cancel_button.click(fn=cancel_processing, inputs=[], outputs=[status_textbox])
    ```
3.  Modify the main `upscale_director_logic` function to handle the cancellation.

    ```python
    # In secourses_app.py, inside the upscale_director_logic function
    # At the very beginning of the function:
    cancellation_manager.reset()
    
    # Wrap the entire existing logic of the function in a try...except...finally block
    try:
        # ... (all existing logic of the function goes here) ...

    except CancelledError:
        logger.warning("Processing was cancelled by user.")
        # Yield a final status message indicating cancellation
        yield (
            # ... (return current state of outputs or None) ...
            gr.update(value="Processing cancelled by user."),
            # ... (other UI updates) ...
        )
    except Exception as e:
        # ... (existing exception handling) ...
    finally:
        # This block runs on success, error, or cancellation
        logger.info("Resetting UI state...")
        cancellation_manager.reset() # Ensure state is reset
        # Return a Gradio update to re-enable the upscale button and hide the cancel button
        yield (
            # ... (final state of outputs) ...
            gr.update(interactive=True), # upscale_button
            gr.update(visible=False),    # cancel_button
            # ... (other UI updates) ...
        )
    ```
4.  Modify the `upscale_button.click` handler to manage button visibility.

    ```python
    # In secourses_app.py
    def upscale_wrapper(*args):
        # First yield: Disable upscale, show cancel
        yield (
            gr.update(interactive=False), # upscale_button
            gr.update(visible=True),      # cancel_button
            # ... (other initial UI updates) ...
        )
        # Then, run the main logic
        for result in upscale_director_logic(...):
            # ... (existing logic to handle intermediate yields) ...
            # The final yield from the `finally` block will reset the buttons
    ```

---

## **Phase 2: Integrating Cancellation into Core Processing Loops**

Now, we'll add checks inside the long-running loops.

### **Step 2.1: `upscaling_core.py`**

1.  Import the cancellation manager: `from .cancellation_manager import cancellation_manager`
2.  In the `run_upscale` function, inside the main processing loop (e.g., `for chunk_info in chunks:` or `for window_iter_idx, i_start_idx in enumerate(window_indices_to_process):`), add the check at the beginning of each iteration.

    ```python
    # In logic/upscaling_core.py, inside run_upscale's main loop
    for chunk_info in chunks:
        cancellation_manager.check_cancel() # Add this line
        # ... rest of the loop logic ...
    ```

### **Step 2.2: `scene_processing_core.py`**

1.  Import the cancellation manager: `from .cancellation_manager import cancellation_manager`
2.  In `process_single_scene`, add the check inside the main loop that processes chunks or windows for that scene.

    ```python
    # In logic/scene_processing_core.py, inside process_single_scene's main loop
    for chunk_info in chunk_boundaries:
        cancellation_manager.check_cancel() # Add this line
        # ... rest of the loop logic ...
    ```

### **Step 2.3: `batch_operations.py`**

1.  Import the cancellation manager: `from .cancellation_manager import cancellation_manager`
2.  In `process_batch_videos`, add the check inside the loop that iterates through the video files.

    ```python
    # In logic/batch_operations.py, inside process_batch_videos
    for i, video_file in enumerate(video_files):
        cancellation_manager.check_cancel() # Add this line
        try:
            # ... rest of the loop logic ...
    ```

---

## **Phase 3: Making Subprocesses Cancellable**

This involves modifying how external tools like FFmpeg and RIFE are called.

### **Step 3.1: Refactor `ffmpeg_utils.py`**

1.  Import the cancellation manager: `from .cancellation_manager import cancellation_manager`
2.  Modify `run_ffmpeg_command` to use `subprocess.Popen` and register the process with the manager.

    ```python
    # In logic/ffmpeg_utils.py
    def run_ffmpeg_command(cmd, desc="ffmpeg command", logger=None, raise_on_error=True):
        # ...
        try:
            process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='ignore')
            cancellation_manager.set_active_process(process)
            
            stdout, stderr = process.communicate()
            
            # Check for cancellation *after* communicate, in case it was cancelled during the run
            cancellation_manager.check_cancel()

            if process.returncode != 0:
                # ... (existing error handling) ...
            
            return True
        # ... (existing exception handling) ...
        finally:
            cancellation_manager.clear_active_process()
    ```

### **Step 3.2: Apply to RIFE and Face Restoration**

1.  Review `rife_interpolation.py` and `face_restoration_utils.py`.
2.  Ensure all `subprocess.run` or `os.system` calls are replaced with the new `run_ffmpeg_command` or a similar `Popen`-based approach that registers with the `cancellation_manager`.

---

## **Phase 4: Interrupting Model Inference**

This is the most direct way to make the model processing itself responsive to cancellation.

### **Step 4.1: Modify Diffusion Callbacks**

1.  In `upscaling_core.py` and `scene_processing_core.py`, locate the diffusion progress callback functions (e.g., `diffusion_callback_for_chunk`).
2.  Add the cancellation check inside these callbacks.

    ```python
    # In logic/upscaling_core.py, inside the diffusion callback
    def diffusion_callback_for_chunk(...):
        cancellation_manager.check_cancel() # Add this line
        # ... rest of the callback logic ...
    ```

This will check for a cancellation request at every step of the diffusion process, providing a highly responsive way to stop the model.

---

## **Phase 5: Final Review and Testing**

1.  **Code Review:** Go through all modified files and ensure the `try...except...finally` blocks are correctly placed, especially in the main UI-facing functions.
2.  **UI State:** Double-check that the "Upscale" and "Cancel" buttons are correctly enabled/disabled at the start and end of every possible execution path (success, error, cancellation).
3.  **Testing Scenarios:**
    *   Cancel during a single video upscale (chunked mode).
    *   Cancel during a single video upscale (context window mode).
    *   Cancel during a batch process (it should stop the current video and not start the next).
    *   Cancel during scene splitting.
    *   Cancel during RIFE processing.
    *   Click "Cancel" multiple times to ensure no errors occur.
    *   Run a process to completion and verify the "Cancel" button becomes hidden and the "Upscale" button is re-enabled.