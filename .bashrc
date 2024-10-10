# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

# User specific environment
if ! [[ "$PATH" =~ "$HOME/.local/bin:$HOME/bin:" ]]
then
    PATH="$HOME/.local/bin:$HOME/bin:$PATH"
fi
export PATH

module load cuda/12.1.1 cmake/3.21.1

# get the install location of cuda by checking the path of nvcc
export CUDA_INSTALL_PATH=$(dirname $(dirname $(which nvcc)))
export CUDACXX=${CUDA_INSTALL_PATH}/bin/nvcc

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=

# User specific aliases and functions

# venvs=~/venvs

# pyact() {
# . $venvs/$1/bin/activate  # commented out by conda initialize
# }

# pylist() {
#         ls $venvs
# }

# pycreate() {
#         python -m venv --system-site-packages $venvs/$1
# }

enqueue() {
    local cmd="$@"
    local current_dir=$(pwd)

    if [ -z "$cmd" ]; then
        echo "No command provided to enqueue."
        return 1
    fi

    echo "cd \"$current_dir\" && $cmd" >> $HOME/.queue
}

pop_queue() {
    local fail_ok=1
    local detach=0
    while [[ "$#" -gt 0 ]]; do
        case "$1" in
            -fn|--fail-not-ok)
                fail_ok=0
                ;;
            -d|--detach)
                detach=1
                ;;
            *)
                echo "Unknown option: $1"
                ;;
        esac
        shift
    done

    local original_dir=$(pwd)  # Capture the original directory
    local cmd=$(head -n 1 $HOME/.queue)
    sed -i '1d' $HOME/.queue # Remove the command from the queue

    if [ -z "$cmd" ]; then
        echo "No commands in queue"
        return 1
    fi

    local status=0
    if [ $detach -eq 1 ]; then
        (
            eval "$cmd";
            status=$?;
            if [ $status -ne 0 ]; then
                echo "Command failed with status $status"
                if [ $fail_ok -eq 0 ]; then
                    { echo "$cmd"; cat $HOME/.queue; } > $HOME/.temp_queue && mv $HOME/.temp_queue $HOME/.queue # Re-add the command to the queue
                fi
            fi
        ) &
    else
        eval "$cmd";
        status=$?;
        if [ $status -ne 0 ]; then
            echo "Command failed with status $status"
            if [ $fail_ok -eq 0 ]; then
                { echo "$cmd"; cat $HOME/.queue; } > $HOME/.temp_queue && mv $HOME/.temp_queue $HOME/.queue # Re-add the command to the queue
            fi
        fi
    fi

    cd "$original_dir"  # Change back to the original directory

    return $status
}

list_queue() {
    cat $HOME/.queue
}

get_node() {
    node_id=$1
    srun -c 8 --mem-per-cpu=1G -G 4 --nodelist=ault$node_id --pty --mail-type=BEGIN -t 04:00:00 bash
}

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/users/lshuhao/mambaforge/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/users/lshuhao/mambaforge/etc/profile.d/conda.sh" ]; then
        . "/users/lshuhao/mambaforge/etc/profile.d/conda.sh"
    else
        export PATH="/users/lshuhao/mambaforge/bin:$PATH"
    fi
fi
unset __conda_setup

if [ -f "/users/lshuhao/mambaforge/etc/profile.d/mamba.sh" ]; then
    . "/users/lshuhao/mambaforge/etc/profile.d/mamba.sh"
fi
# <<< conda initialize <<<

export VCPKG_ROOT=~/vcpkg
export PATH=$VCPKG_ROOT:$PATH
export CUTLASS_DIR=~/cutlass
