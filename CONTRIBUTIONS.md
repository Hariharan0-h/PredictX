# Internship Contribution Log

**Intern Name:** Vanshika  
**Project:** Contextual Predictive Maintenance (IoT Edge AI)  
**Company:** Infotact Solutions & Co.  

---

## Week 1: Environment Setup & Telemetry Ingestion
- **Local Setup:** Successfully– cloned the industrial predictive maintenance repository to the local environment using Git.
- **Environment Verification:** Configured VS Code and verified the repository structure (`data`, `notebooks`, `src`, `tests`).
- **Dataset Study:** Initiated the documentation review for the `AI4I Predictive Maintenance Dataset` to understand rolling statistical feature requirements.
- **Current Status:** Workspace is fully operational and prepared for time-series signal processing.
-
### ## Week 2: Code Documentation and Enhancement
- *Tasks Completed:* Successfully added professional docstrings to improve code readability and documentation standards across multiple core modules.
- *Files Modified:* 
  - src/ingest.py (Added module and function level docstrings)
  - src/features.py (Documented feature engineering functions)
  - src/predict.py (Documented threshold loading and inference logic)
  - src/train.py (Documented pipeline building and training execution)
- *Commit Target:* Completed the weekly goal of 10 structural and documentation commits.
- *Current Status:* Core codebase documentation completed and synced with remote repository.

## Week 3: Model Evaluation Pipeline & Metrics Logging
- **Tasks Completed:** Successfully built and integrated comprehensive model evaluation logging, noise injection handling, and precision-recall threshold optimization.
- *Files Modified:*
    - src/evaluate.py (Implemented complete evaluation pipeline, noise scaling, and performance tracking)
    - custom_logging/logger.py (Fixed AttributeError by attaching LOGS_DIR to the logger object)
- *Commit Target:* Completed structural validation and robust logging enhancements for the evaluation codebase.
- *Current Status:* Core evaluation pipeline fully functional, verified with precise threshold outputs, and synchronized with remote repository.
-