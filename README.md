<h2 align="center">
    <a href="https://computer-use-ootb.github.io">
        <img src="./assets/ootb_logo.png" alt="Logo" style="display: block; margin: 0 auto; filter: invert(1) brightness(2);">
    </a>
</h2>


<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.</h5>

<h5 align=center>

[![arXiv](https://img.shields.io/badge/Arxiv-2411.10323-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2411.10323)
[![Project Page](https://img.shields.io/badge/Project_Page-GUI_Agent-blue)](https://computer-use-ootb.github.io)


</h5>

## <img src="./assets/ootb_icon.png" alt="Star" style="height:25px; vertical-align:middle; filter: invert(1) brightness(2);">  Overview
**Computer Use <span style="color:rgb(106, 158, 210)">O</span><span style="color:rgb(111, 163, 82)">O</span><span style="color:rgb(209, 100, 94)">T</span><span style="color:rgb(238, 171, 106)">B</span>**<img src="./assets/ootb_icon.png" alt="Star" style="height:20px; vertical-align:middle; filter: invert(1) brightness(2);"> is an out-of-the-box (OOTB) solution for Desktop GUI Agent, including API-based (**Claude 3.5 Computer Use**) and locally-running models (**<span style="color:rgb(106, 158, 210)">S</span><span style="color:rgb(111, 163, 82)">h</span><span style="color:rgb(209, 100, 94)">o</span><span style="color:rgb(238, 171, 106)">w</span>UI**, **UI-TARS**). 

**No Docker** is required, and it supports both **Windows** and **macOS**. OOTB provides a user-friendly interface based on Gradio.🎨

Visit our study on GUI Agent of Claude 3.5 Computer Use [[project page]](https://computer-use-ootb.github.io). 🌐

## Update
- **[2025/02/08]** We've added the support for [**UI-TARS**](https://github.com/bytedance/UI-TARS). Follow [Cloud Deployment](https://github.com/bytedance/UI-TARS?tab=readme-ov-file#cloud-deployment) or [VLLM deployment](https://github.com/bytedance/UI-TARS?tab=readme-ov-file#local-deployment-vllm) to implement UI-TARS and run it locally in OOTB.
- **Major Update! [2024/12/04]** **Local Run🔥** is now live! Say hello to [**<span style="color:rgb(106, 158, 210)">S</span><span style="color:rgb(111, 163, 82)">h</span><span style="color:rgb(209, 100, 94)">o</span><span style="color:rgb(238, 171, 106)">w</span>UI**](https://github.com/showlab/ShowUI), an open-source 2B vision-language-action (VLA) model for GUI Agent. Now compatible with `"gpt-4o + ShowUI" (~200x cheaper)`*  & `"Qwen2-VL + ShowUI" (~30x cheaper)`* for only few cents for each task💰! <span style="color: grey; font-size: small;">*compared to Claude Computer Use</span>.
- **[2024/11/20]** We've added some examples to help you get hands-on experience with Claude 3.5 Computer Use.
- **[2024/11/19]** Forget about the single-display limit set by Anthropic - you can now use **multiple displays** 🎉!
- **[2024/11/18]** We've released a deep analysis of Claude 3.5 Computer Use: [https://arxiv.org/abs/2411.10323](https://arxiv.org/abs/2411.10323).
- **[2024/11/11]** Forget about the low-resolution display limit set by Anthropic — you can now use *any resolution you like* and still keep the **screenshot token cost low** 🎉!
- **[2024/11/11]** Now both **Windows** and **macOS** platforms are supported 🎉!
- **[2024/10/25]** Now you can **Remotely Control** your computer 💻 through your mobile device 📱 — **No Mobile App Installation** required! Give it a try and have fun 🎉.


## Demo Video

https://github.com/user-attachments/assets/f50b7611-2350-4712-af9e-3d31e30020ee

<div style="display: flex; justify-content: space-around;">
  <a href="https://youtu.be/Ychd-t24HZw" target="_blank" style="margin-right: 10px;">
    <img src="https://img.youtube.com/vi/Ychd-t24HZw/maxresdefault.jpg" alt="Watch the video" width="48%">
  </a>
  <a href="https://youtu.be/cvgPBazxLFM" target="_blank">
    <img src="https://img.youtube.com/vi/cvgPBazxLFM/maxresdefault.jpg" alt="Watch the video" width="48%">
  </a>
</div>


## 🚀 Getting Started

### 0. Prerequisites
- Instal Miniconda on your system through this [link](https://www.anaconda.com/download?utm_source=anacondadocs&utm_medium=documentation&utm_campaign=download&utm_content=topnavalldocs). (**Python Version: >= 3.12**).
- Hardware Requirements (optional, for ShowUI local-run):
    - **Windows (CUDA-enabled):** A compatible NVIDIA GPU with CUDA support, >=6GB GPU memory
    - **macOS (Apple Silicon):** M1 chip (or newer), >=16GB unified RAM


### 1. Clone the Repository 📂
Open the Conda Terminal. (After installation Of Miniconda, it will appear in the Start menu.)
Run the following command on **Conda Terminal**.
```bash
git clone https://github.com/showlab/computer_use_ootb.git
cd computer_use_ootb
```

### 2. Use the Setup Assistant (Recommended) 🤖

Run the interactive helper to check your environment, install dependencies, download optional local models, and generate API key templates:

```bash
python install_tools/setup_ootb.py
```

The assistant can:
- verify whether you're inside a Conda/venv environment;
- install `requirements.txt` using your current Python interpreter;
- optionally download ShowUI models (full precision or AWQ 4-bit) or review UI-TARS/Qwen deployment tips;
- create `.env` or `api_keys.json` files with placeholders so you can add API keys later;
- print the next steps (e.g., `python app.py`, default port `7860`).

You can rerun the script at any time—every step is optional and idempotent. If you prefer manual setup, continue with Sections 3.1–3.4 below.

### 3. Manual Setup (if you skipped the assistant)

#### 3.1 Install Dependencies 🔧

If you skipped the assistant, install the Python requirements yourself:

```bash
pip install -r requirements.txt
```

#### 3.2 (Optional) Get Prepared for **<span style="color:rgb(106, 158, 210)">S</span><span style="color:rgb(111, 163, 82)">h</span><span style="color:rgb(209, 100, 94)">o</span><span style="color:rgb(238, 171, 106)">w</span>UI** Local-Run

1. Download all files of the ShowUI-2B model via the following command. Ensure the `ShowUI-2B` folder is under the `computer_use_ootb` folder.

    ```python
    python install_tools/install_showui.py
    ```

2. (Optional, CUDA only) Download the AWQ 4-bit weights:

    ```python
    python install_tools/install_showui-awq-4bit.py
    ```

3. Make sure to install the correct GPU version of PyTorch (CUDA, MPS, etc.) on your machine. See [install guide and verification](https://pytorch.org/get-started/locally/).

4. Get API Keys for [GPT-4o](https://platform.openai.com/docs/quickstart) or [Qwen-VL](https://help.aliyun.com/zh/dashscope/developer-reference/acquisition-and-configuration-of-api-key). For mainland China users, Qwen API free trial for first 1 mil tokens is [available](https://help.aliyun.com/zh/dashscope/developer-reference/tongyi-qianwen-vl-plus-api).

#### 3.3 (Optional) Get Prepared for **UI-TARS** Local-Run

1. Follow [Cloud Deployment](https://github.com/bytedance/UI-TARS?tab=readme-ov-file#cloud-deployment) or [VLLM deployment](https://github.com/bytedance/UI-TARS?tab=readme-ov-file#local-deployment-vllm) guides to deploy your UI-TARS server.

2. Test your UI-TARS sever with the script `.\install_tools\test_ui-tars_server.py`.

#### 3.4 (Optional) Deploy the Qwen Planner on a Remote Server

1. Clone this project on your SSH server.

2. Start the planner bridge:

    ```bash
    python computer_use_demo/remote_inference.py --host 0.0.0.0 --port 8000
    ```

### 4. Start the Interface ▶️

**Start the OOTB interface:**
```bash
python app.py
```
If you successfully start the interface, you will see two URLs in the terminal:
```bash
* Running on local URL:  http://127.0.0.1:7860
* Running on public URL: https://xxxxxxxxxxxxxxxx.gradio.live (Do not share this link with others, or they will be able to control your computer.)
```


> <u>For convenience</u>, we recommend running one or more of the following command to set API keys to the environment variables before starting the interface. Then you don’t need to manually pass the keys each run. On Windows Powershell (via the `set` command if on cmd): 
> ```bash
> $env:ANTHROPIC_API_KEY="sk-xxxxx" (Replace with your own key)
> $env:QWEN_API_KEY="sk-xxxxx"
> $env:OPENAI_API_KEY="sk-xxxxx"
> $env:GEMINI_API_KEY="sk-xxxxx"
> ```
> On macOS/Linux, replace `$env:ANTHROPIC_API_KEY` with `export ANTHROPIC_API_KEY` in the above command. 


### 5. Control Your Computer with Any Device can Access the Internet
- **Computer to be controlled**: The one installed software.
- **Device Send Command**: The one opens the website.
  
Open the website at http://localhost:7860/ (if you're controlling the computer itself) or https://xxxxxxxxxxxxxxxxx.gradio.live in your mobile browser for remote control.

Enter the Anthropic API key (you can obtain it through this [website](https://console.anthropic.com/settings/keys)), then give commands to let the AI perform your tasks.

### ShowUI Advanced Settings

We provide a 4-bit quantized ShowUI-2B model for cost-efficient inference (currently **only support CUDA devices**). To download the 4-bit quantized ShowUI-2B model:
```
python install_tools/install_showui-awq-4bit.py
```
Then, enable the quantized setting in the 'ShowUI Advanced Settings' dropdown menu.

Besides, we also provide a slider to quickly adjust the `max_pixel` parameter in the ShowUI model. This controls the visual input size of the model and greatly affects the memory and inference speed.

### Troubleshooting Dependency & Permission Issues

| Symptom | Likely Cause | Suggested Fix |
| --- | --- | --- |
| `pip` cannot write to site-packages | Missing admin rights / system Python | Activate a Conda env (`conda create -n ootb python=3.11`) and rerun `python install_tools/setup_ootb.py`. |
| SSL or proxy errors when installing packages | Corporate network blocks PyPI | Configure your proxy in `pip.ini`/`.pip/pip.conf` or download wheels manually, then rerun the assistant. |
| CUDA libraries not found while starting ShowUI | PyTorch GPU build missing | Reinstall PyTorch with the correct CUDA/MPS build: [install guide](https://pytorch.org/get-started/locally/). |
| `PermissionError` when creating `.env` or model folders | Script executed from read-only directory | Move the repo to a writable location (e.g., `~/computer_use_ootb`) or rerun the script with elevated permissions. |

## 📊 GUI Agent Model Zoo

Now, OOTB supports customizing the GUI Agent via the following models:

- **Unified Model**: Unified planner & actor, can both make the high-level planning and take the low-level control.
- **Planner**: General-purpose LLMs, for handling the high-level planning and decision-making.
- **Actor**: Vision-language-action models, for handling the low-level control and action command generation.


<div align="center">
  <b>Supported GUI Agent Models, OOTB</b>

</div>
<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>[API] Unified Model</b>
      </td>
      <td>
        <b>[API] Planner</b>
      </td>
      <td>
        <b>[Local] Planner</b>
      </td>
      <td>
        <b>[API] Actor</b>
      </td>
      <td>
        <b>[Local] Actor</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
            <li><a href="">Claude 3.5 Sonnet</a></li>
      </ul>
      </td>
      <td>
        <ul>
          <li><a href="">GPT-4o</a></li>
          <li><a href="">Qwen2-VL-Max</a></li>
          <li><a href="">Qwen2-VL-2B(ssh)</a></li>
          <li><a href="">Qwen2-VL-7B(ssh)</a></li>
          <li><a href="">Qwen2.5-VL-7B(ssh)</a></li>
          <li><a href="">Deepseek V3 (soon)</a></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="">Qwen2-VL-2B</a></li>
          <li><a href="">Qwen2-VL-7B</a></li>
        </ul>
      </td>
        <td>
        <ul>
          <li><a href="https://github.com/showlab/ShowUI">ShowUI</a></li>
          <li><a href="https://huggingface.co/bytedance-research/UI-TARS-7B-DPO">UI-TARS-7B/72B-DPO (soon)</a></li> 
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="https://github.com/showlab/ShowUI">ShowUI</a></li>
          <li><a href="https://huggingface.co/bytedance-research/UI-TARS-7B-DPO">UI-TARS-7B/72B-DPO</a></li>
        </ul>
      </td>
    </tr>
</td>
</table>

> where [API] models are based on API calling the LLMs that can inference remotely, 
and [Local] models can use your own device that inferences locally with no API costs.



## 🖥️ Supported Systems
- **Windows** (Claude ✅, ShowUI ✅)
- **macOS** (Claude ✅, ShowUI ✅)

## 👓 OOTB Iterface
<div style="display: flex; align-items: center; gap: 10px;">
  <figure style="text-align: center;">
    <img src="./assets/gradio_interface.png" alt="Desktop Interface" style="width: auto; object-fit: contain;">
  </figure>
</div>


## ⚠️ Risks
- **Potential Dangerous Operations by the Model**: The models' performance is still limited and may generate unintended or potentially harmful outputs. Recommend continuously monitoring the AI's actions. 
- **Cost Control**: Each task may cost a few dollars for Claude 3.5 Computer Use.💸

## 📅 Roadmap
- [ ] **Explore available features**
  - [ ] The Claude API seems to be unstable when solving tasks. We are investigating the reasons: resolutions, types of actions required, os platforms, or planning mechanisms. Welcome any thoughts or comments on it.
- [ ] **Interface Design**
  - [x] **Support for Gradio** ✨
  - [ ] **Simpler Installation**
  - [ ] **More Features**... 🚀
- [ ] **Platform**
  - [x] **Windows**
  - [x] **macOS**
  - [x] **Mobile** (Send command)
  - [ ] **Mobile** (Be controlled)
- [ ] **Support for More MLLMs**
  - [x] **Claude 3.5 Sonnet** 🎵
  - [x] **GPT-4o**
  - [x] **Qwen2-VL**
  - [ ] **Local MLLMs**
  - [ ] ...
- [ ] **Improved Prompting Strategy**
  - [ ] Optimize prompts for cost-efficiency. 💡
- [x] **Improved Inference Speed**
  - [x] Support int4 Quantization.

## Join Discussion
Welcome to discuss with us and continuously improve the user experience of Computer Use - OOTB. Reach us using this [**Discord Channel**](https://discord.gg/vMMJTSew37) or the WeChat QR code below!

<div style="display: flex; flex-direction: row; justify-content: space-around;">

<!-- <img src="./assets/wechat_2.jpg" alt="gradio_interface" width="30%"> -->
<img src="./assets/wechat_3.jpg" alt="gradio_interface" width="30%">

</div>

<div style="height: 30px;"></div>

<hr>
<a href="https://computer-use-ootb.github.io">
<img src="./assets/ootb_logo.png" alt="Logo" width="30%" style="display: block; margin: 0 auto; filter: invert(1) brightness(2);">
</a>



