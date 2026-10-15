"""Pure acceptance and continuation policy for the explicitly requested long run."""
import math


def strict_target(spectrum, wavelengths, *, target_tx=.80, target_ty=.10):
    expected = sorted(float(w) for w in wavelengths)
    return bool(expected) and sorted(float(r["wavelength_nm"]) for r in spectrum) == expected and all(
        math.isfinite(r["tx"]) and math.isfinite(r["ty"])
        and target_tx < r["tx"] <= 1.001 and 0 <= r["ty"] < target_ty for r in spectrum
    )


def recycle_due(control, step, updates, max_updates):
    """Opt-in, bounded process recycling after a committed optimizer update."""
    if not isinstance(control, dict) or control.get("enabled") is not True:
        return False
    interval, through = control.get("every_updates"), control.get("through_step")
    if type(interval) is not int or interval < 1 or type(through) is not int or through < 0:
        return False
    return (0 <= step <= through and updates == step + 1 and updates % interval == 0
            and (max_updates is None or updates < max_updates))


def full_restart_checkpoint(control, row, status, process, request, managed_stop):
    """Only explicitly marked ordinary recycling can omit a historical tensor copy."""
    if not all(isinstance(item, dict) for item in (control, row, status, process, request)):
        return True
    iteration = request.get("iteration")
    if not isinstance(iteration, dict):
        return True
    step = row.get("step")
    routine = (
        control.get("archive_policy") == "binary_milestones"
        and managed_stop
        and request.get("kind") == "worker_recycle"
        and type(process.get("runner_pid")) is int and process["runner_pid"] > 0
        and request.get("runner_pid") == process["runner_pid"]
        and type(step) is int and step >= 0
        and iteration.get("step") == step
        and status.get("status") == "checkpoint_recycle"
        and status.get("evaluated_step") == step
        and status.get("updates_completed") == step + 1
        and row.get("update_applied") is True
    )
    return not routine or row.get("bin_fom") is not None


def set_option(command, key, value):
    command = list(command)
    if key in command:
        command[command.index(key)+1] = str(value)
    else:
        command.extend((key, str(value)))
    return command


def continuation_command(command, config, updates):
    command = list(command)
    for key in ("--init-state", "--init-global-step"):
        if key in command:
            index = command.index(key)
            del command[index:index+2]
    if updates >= config["max_updates"]:
        command = set_option(command, "--max-updates", updates + 100)
        # Do not restart the cosine decay or raise the learning rate at a budget extension.
        rate = config.get("lr_final") if config.get("lr_final") is not None else config["lr"]
        command = set_option(command, "--lr", rate)
        command = set_option(command, "--lr-final", rate)
    return command
