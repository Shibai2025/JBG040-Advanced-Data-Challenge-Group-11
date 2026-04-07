import subprocess
import sys


def run_script(step_name:str, command_list:list[str]) -> None:
    print("\n" + "=" * 100)
    print(f"Starting step: {step_name}")
    print(f"COMMAND: {' '.join(command_list)}")
    print("=" * 100)

    result = subprocess.run(command_list)
    if result.returncode != 0:
        print(f"Step failed: {step_name} (exit code {result.returncode})")
        sys.exit(1)


if __name__ == "__main__":
    python_exe = sys.executable

    run_script(
        step_name="Sample Representatives",
        command_list=[python_exe, "find_representative_samples_test_clip_limit.py"]
    )

    run_script(
        step_name="Initial Clip Limit Testing",
        command_list=[python_exe, "test_clip_limit_first_trial.py"]
    )


    run_script(
        step_name="Clip Limit Testing for the Lungs",
        command_list=[python_exe, "test_clip_limit_lungs.py"]
    )

    run_script(
        step_name="Creating Cropped Image Dataset",
        command_list=[python_exe, "image_processing_with_computer_vision.py"]
    )

    run_script(
        step_name="Clip Limit Verification",
        command_list=[python_exe, "lighting_verification.py"]
    )

    run_script(
        step_name="Cropping Verification",
        command_list=[python_exe, "verification_of_cropping.py"]
    )

    print("All Computer Vision Files Ran Successfully.")