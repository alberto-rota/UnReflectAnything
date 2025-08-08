# -------------------------------------------------------------------------------------------------#

""" Copyright (c) 2024 Asensus Surgical """

""" Code Developed by: Alberto Rota """
""" Supervision: Uriya Levy, Gal Weizman, Stefano Pomati """

# -------------------------------------------------------------------------------------------------#

import io
import json
import os
import re
import socket
import subprocess
from contextlib import redirect_stdout
import paramiko
from rich import print
import gc
import torch
import inspect
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout


def get_hostname() -> str:

    hostname = socket.gethostname()
    if "alberto-vm" in hostname:
        return "AsensusVM"
    elif "rk018445" in hostname:
        return "MUFASA"


def sftp_transfer(local_file_path: str, remote_file_path: str) -> None:
    """
    Transfers a file from a local directory to a remote SFTP server.

    Args:
        local_file_path (str): The path to the local file to be transferred.
        remote_file_path (str): The destination path on the remote SFTP server.

    Returns:
        None
    """

    # Load the SFTP credentials from a JSON file
    try:
        with open("/home/alberto/MONO3D/secrets/keys.json") as f:
            sftp_credentials = json.load(f)

        # Extract SFTP credentials
        sftp_host = sftp_credentials["sftp_host"]
        sftp_port = sftp_credentials["sftp_port"]
        sftp_username = sftp_credentials["sftp_username"]
        sftp_password = sftp_credentials["sftp_password"]

        # Initialize the SFTP client using the provided credentials
        transport = paramiko.Transport((sftp_host, sftp_port))
        transport.connect(username=sftp_username, password=sftp_password)
        sftp = paramiko.SFTPClient.from_transport(transport)

        # Upload the file from the local path to the remote path
        sftp.put(local_file_path, remote_file_path)

        # Close the SFTP connection
        sftp.close()
        transport.close()
        print(f" [green]-->OK-->[/green] Saved to NAS:/{remote_file_path}")

    except Exception as e:
        print(f" [red]-->ERROR-->[/red] Cound not upload file to NAS : ", end="")
        print(e)
        return


def titlescreen() -> None:
    """
    Prints the title screen from a text file.
    """
    with open("assets/banner_giant.txt", "r") as f:
        print(f"[white]{f.read()}[/white]")


def check_rerun_output(func, *args, **kwargs):
    # Create a buffer to capture output
    buffer = io.StringIO()

    # Run the function while capturing its output
    with redirect_stdout(buffer):
        func(*args, **kwargs)

    # Get the captured output
    output = buffer.getvalue()

    # Define the pattern to search for
    pattern = r"WARN  re_sdk_comms"

    # Check if the output matches the pattern
    if re.search(pattern, output):
        print(
            "[RERUN] >> [orange3]Cannot communicate with server. Will not log[/orange3]"
        )
        return False
    else:
        # If no pattern match, print the original output
        print("[RERUN] >> [green]Connection established[/green]")
        return True


def detect_aval_cpus():
    """
    Detects the number of available CPUs.
    """
    try:
        currentjobid = os.environ["SLURM_JOB_ID"]
        currentjobid = int(currentjobid)
        command = f"squeue --Format=JobID,cpus-per-task | grep {currentjobid}"
        # Run the command as a subprocess and capture the output
        output = subprocess.check_output(command, shell=True)[5:-4].replace(b" ", b"")
        cpus = output.decode("utf-8")
        cpus2 = len(os.sched_getaffinity(0))
        cpus = min(int(cpus), cpus2)
    except:
        cpus = 1  # os.cpu_count()
    return cpus


import gc
import torch
import inspect
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text


def is_all_tensors_on_gpu(container):
    """Check if all items in a container are GPU tensors."""
    if isinstance(container, dict):
        return all(
            isinstance(v, torch.Tensor) and v.is_cuda for v in container.values()
        )
    elif isinstance(container, (list, tuple)):
        return all(isinstance(v, torch.Tensor) and v.is_cuda for v in container)
    return False


import gc
import torch
import inspect
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from types import FrameType


def is_all_tensors_on_gpu(container):
    """Check if all items in a container are GPU tensors."""
    if isinstance(container, dict):
        return all(
            isinstance(v, torch.Tensor) and v.is_cuda for v in container.values()
        )
    elif isinstance(container, (list, tuple)):
        return all(isinstance(v, torch.Tensor) and v.is_cuda for v in container)
    return False


def gpuClean(frame_up=1, exclude_vars=None, verbose=False):
    """
    Automatically detects and frees GPU memory by cleaning up tensor variables from parent scopes.

    Parameters:
    frame_up (int): How many frames up to clean (1 = parent function, 2 = grandparent, etc.)
    exclude_vars (list, optional): List of variable names to exclude from cleanup
    verbose (bool): Whether to print information about cleaned variables
    """
    console = Console()

    frame = inspect.currentframe()
    for _ in range(frame_up):
        frame = frame.f_back

    if frame is None:
        console.print("[red]Warning: Could not access the specified parent frame[/red]")
        return 0, 0

    # Get both locals and globals from the frame
    local_vars = frame.f_locals
    global_vars = frame.f_globals
    caller_name = frame.f_code.co_name

    exclude_vars = set(exclude_vars or [])
    freed_count = 0
    total_memory_freed = 0

    if verbose:
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Variable Name", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Shape/Keys", style="yellow")
        table.add_column("Memory", style="red")
        console.print(
            f"\n[bold blue]Starting cleanup in scope: [cyan]{caller_name}()[/cyan][/bold blue]"
        )

    def delete_tensor(tensor):
        """Helper function to properly delete a tensor."""
        if hasattr(tensor, "grad") and tensor.grad is not None:
            tensor.grad.data.zero_()
            tensor.grad.detach_()
            del tensor.grad

        tensor.detach_()
        if tensor.is_cuda:
            tensor.cpu()
        del tensor

    # Store variables to delete
    to_delete = []

    # First pass: identify what to delete and log it
    for name, var in local_vars.items():
        if name in exclude_vars:
            continue

        if isinstance(var, torch.Tensor) and var.is_cuda:
            try:
                size_mb = var.element_size() * var.nelement() / (1024 * 1024)
                total_memory_freed += size_mb

                if verbose:
                    size_str = (
                        f"{size_mb:.2f}MB"
                        if size_mb < 1024
                        else f"{(size_mb/1024):.2f}GB"
                    )
                    table.add_row(name, "Tensor", str(list(var.shape)), size_str)

                to_delete.append((name, "tensor", var))
                freed_count += 1
            except Exception as e:
                if verbose:
                    console.print(
                        f"[red]Error examining tensor '{name}': {str(e)}[/red]"
                    )

        elif isinstance(var, (dict, list, tuple)):
            if is_all_tensors_on_gpu(var):
                try:
                    if isinstance(var, dict):
                        total_size_mb = sum(
                            t.element_size() * t.nelement() / (1024 * 1024)
                            for t in var.values()
                        )
                        keys_info = f"Keys: {list(var.keys())}"
                        container_type = "Dict of Tensors"
                    else:
                        total_size_mb = sum(
                            t.element_size() * t.nelement() / (1024 * 1024) for t in var
                        )
                        keys_info = f"Length: {len(var)}"
                        container_type = f"{type(var).__name__} of Tensors"

                    total_memory_freed += total_size_mb

                    if verbose:
                        size_str = (
                            f"{total_size_mb:.2f}MB"
                            if total_size_mb < 1024
                            else f"{(total_size_mb/1024):.2f}GB"
                        )
                        table.add_row(name, container_type, keys_info, size_str)

                    to_delete.append((name, "container", var))
                    freed_count += 1

                except Exception as e:
                    if verbose:
                        console.print(
                            f"[red]Error examining container '{name}': {str(e)}[/red]"
                        )

    # Second pass: actually delete everything
    for name, var_type, var in to_delete:
        try:
            if var_type == "tensor":
                delete_tensor(var)
            elif var_type == "container":
                if isinstance(var, dict):
                    for k, v in var.items():
                        if isinstance(v, torch.Tensor):
                            delete_tensor(v)
                elif isinstance(var, (list, tuple)):
                    for v in var:
                        if isinstance(v, torch.Tensor):
                            delete_tensor(v)

            # Remove from both locals and globals
            if name in local_vars:
                local_vars[name] = None
                del local_vars[name]
            if name in global_vars:
                global_vars[name] = None
                del global_vars[name]

        except Exception as e:
            if verbose:
                console.print(f"[red]Error deleting '{name}': {str(e)}[/red]")

    # Force update of frame locals and globals
    frame.f_locals.update(local_vars)

    # Force garbage collection multiple times to ensure cleanup
    for _ in range(3):
        gc.collect()
    torch.cuda.empty_cache()

    if verbose and freed_count > 0:
        console.print(table)

        summary_text = []
        summary_text.append(f"Tensors freed: {freed_count}")

        if total_memory_freed >= 1024:
            summary_text.append(
                f"Total memory freed: [bold green]{total_memory_freed/1024:.2f}GB[/bold green]"
            )
        else:
            summary_text.append(
                f"Total memory freed: [bold green]{total_memory_freed:.2f}MB[/bold green]"
            )

        current_allocated = torch.cuda.memory_allocated() / 1024 / 1024
        current_reserved = torch.cuda.memory_reserved() / 1024 / 1024

        if current_allocated >= 1024:
            summary_text.append(
                f"Current GPU allocated: [bold yellow]{current_allocated/1024:.2f}GB[/bold yellow]"
            )
        else:
            summary_text.append(
                f"Current GPU allocated: [bold yellow]{current_allocated:.2f}MB[/bold yellow]"
            )

        if current_reserved >= 1024:
            summary_text.append(
                f"Current GPU reserved: [bold red]{current_reserved/1024:.2f}GB[/bold red]"
            )
        else:
            summary_text.append(
                f"Current GPU reserved: [bold red]{current_reserved:.2f}MB[/bold red]"
            )

        if exclude_vars:
            summary_text.append(
                f"\nExcluded variables: [dim]{', '.join(exclude_vars)}[/dim]"
            )

        console.print(
            Panel(
                "\n".join(summary_text),
                title="[bold]Cleanup Summary[/bold]",
                border_style="blue",
            )
        )
    elif verbose:
        console.print("[yellow]No tensors were cleaned up[/yellow]")

    return freed_count, total_memory_freed
