.PHONY: all clean submit clean_submit submit_ocr clean_submit_ocr

submit:
	sbatch --export=CATALOGUE_NAME=$(name) resources/slurm_jobs/schedule_run_with_extract.sh
	@echo "Submitting job for catalogue $(name) to SLURM"
	@echo "To check the status of the job, use: squeue -u \$\USER"
	@echo "To cancel the job, use: scancel <job_id>"
	@echo "To check the tail of the logs, use: tail -f slurm-<job_id>.out"

submit_ocr:
	sbatch --export=CATALOGUE_NAME=$(name) resources/slurm_jobs/schedule_run_ocr.sh
	@echo "Submitting job for catalogue $(name) to SLURM"
	@echo "To check the status of the job, use: squeue -u \$\USER"
	@echo "To cancel the job, use: scancel <job_id>"
	@echo "To check the tail of the logs, use: tail -f slurm-<job_id>.out"
clean:
	@echo "Cleaning up SLURM output files"
	rm -rf slurm-*.out


clean_submit:
	@echo "========================================================================"
	@$(MAKE) clean
	@echo "========================================================================"
	@$(MAKE) submit name=lightfootcat
	@echo "========================================================================"
	@$(MAKE) submit name=hanbury
	@echo "========================================================================"

clean_submit_ocr:
	@echo "========================================================================"
	@$(MAKE) clean
	@echo "========================================================================"
	@$(MAKE) submit_ocr name=lightfootcat
	@echo "========================================================================"
	@$(MAKE) submit_ocr name=hanbury
	@echo "========================================================================"