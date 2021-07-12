## Installing rbfdqn
- Cloned from https://github.com/kavosh8/RBFDQN_pytorch (b2723aaacd34891a91486435bfdd2ca688bad8a1)
- Missing gym
    - `pip install gym`
- Missing cpprb
    - `pip install cpprb`
        - Fails because of clang issue. Need to use gcc/g++
    - `brew gcc llvm`
    - `source rbfdqn/gcc_export`
    - gcc_export:
        ```
        export CC=/usr/local/bin/gcc-10
        export CXX=/usr/local/bin/g++-10
        export CPP=/usr/local/bin/cpp-10
        export LD=/usr/local/bin/gcc-10
        alias c++=/usr/local/bin/c++-10
        alias g++=/usr/local/bin/g++-10
        alias gcc=/usr/local/bin/gcc-10
        alias cpp=/usr/local/bin/cpp-10
        alias ld=/usr/local/bin/gcc-10
        alias cc=/usr/local/bin/gcc-10
        ```
    - Re-run `pip install cpprb`
    - For CCV:
        ```
        module load gcc/10.2
        pip install git+git://github.com/ymd-h/cpprb.git
        ```
- Test as follows:
    - Pendulum-v0: `python -m rbfdqn 00 0`
    - LunarLanderContinuous-v2: `python -m rbfdqn 10 0`
    - BipedalWalker-v3: `python -m rbfdqn 20 0`
    - Hopper-v3: `python -m rbfdqn 30 0`
    - HalfCheetah-v3: `python -m rbfdqn 40 0`
    - Ant-v3: `python -m rbfdqn 30 0`

## Installing mujoco
- First install Mujoco 1.5 and 2.0 to ~/.mujoco
- Then hook it up to python
    - `pip install "mujoco-py<2.1,>=2.0"`
- Test with python3:
    ```python
    import mujoco_py
    import os
    mj_path, _ = mujoco_py.utils.discover_mujoco()
    xml_path = os.path.join(mj_path, 'model', 'humanoid.xml')
    model = mujoco_py.load_model_from_path(xml_path)
    sim = mujoco_py.MjSim(model)

    print(sim.data.qpos)
    # [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

    sim.step()
    print(sim.data.qpos)
    # [-2.09531783e-19  2.72130735e-05  6.14480786e-22 -3.45474715e-06
    #   7.42993721e-06 -1.40711141e-04 -3.04253586e-04 -2.07559344e-04
    #   8.50646247e-05 -3.45474715e-06  7.42993721e-06 -1.40711141e-04
    #  -3.04253586e-04 -2.07559344e-04 -8.50646247e-05  1.11317030e-04
    #  -7.03465386e-05 -2.22862221e-05 -1.11317030e-04  7.03465386e-05
    #  -2.22862221e-05]
    ```
- Expected to find MuJoCo here: ~/.mujoco/mujoco200
    - `cd ~/.mujoco && ln -s mujoco200_macos/ mujoco200`
- May need to wait for it to rebuild mujoco, which takes a minute or two.

## Installing dm_control
- `pip install dm-control`
- Test with python:
    - `python test/test_dm2gym.py`

## Installing dm2gym
- Current 0.2.0 release on PyPI has broken rendering; install from github
    - `pip install git+git://github.com/zuoxingdong/dm2gym.git`
- Needs cv2
    - `pip install opencv-python`
- Test with python:
    - `python test/test_dm2gym.py`

## Running on CCV
- See note above about installing cpprb
- Mujoco needs an OpenGL backend:
    - EGL (headless, hardware-accelerated)
    - GLFW (windowed, hardware-accelerated)
    - OSMesa (purely software-based)

-----
## Singularity
- https://sylabs.io/guides/3.6/admin-guide/installation.html#mac
```bash
brew install --cask virtualbox
brew install --cask vagrant
brew install --cask vagrant-manager
mkdir vm-singularity && cd vm-singularity
export VM=sylabs/singularity-3.6-ubuntu-bionic64 && \
    vagrant init $VM && \
    vagrant up && \
    vagrant ssh
exit
vagrant pluging install vagrant-scp
vagrant scp headless.def :~/
vagrant ssh
sudo vagrant build headless.img headless.def
exit
vagrant scp :~/headless.sif .
scp headless.sif cluster:~/
```

On cluster:
```
singularity shell --nv -B /gpfs/scratch,/gpfs/data ~/headless.sif
```

## Running on CCV with singularity

0. Ensure singularity image ~/headless.sif exists.
1. Update python packages in normal virtualenv.
2. Update pipenv packages inside singularity.
    ```
    singularity exec -B /gpfs/scratch,/gpfs/data ~/headless.sif sh -c "PIPENV_IGNORE_VIRTUALENVS=1 pipenv install -r requirements.txt --skip-lock --ignore-pipfile"
    ```
3. Configure onager:
    .onager/config:
    ```
    [slurm]
    header = singularity exec --nv -B /gpfs/scratch,/gpfs/data ~/headless.sif sh -c ". venv/bin/activate && \
    footer = "
    ```
4. When calling prelaunch, prefix the command as follows:
    ```
    PIPENV_IGNORE_VIRTUALENVS=1 xvfb-run -a pipenv run python -m some.module
    ```
