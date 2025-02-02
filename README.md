---
title: SmolLM2 Text Generator
emoji: 🔥
colorFrom: indigo
colorTo: gray
sdk: docker
pinned: false
---


# SmolLM2 GPT Text Generator

This project is a web-based application for generating text continuations using GPT-based models. Users can input text through a simple UI, and the application will respond with a continuation generated by the underlying GPT model. Model is based on the SmolLM2 architecture. It is a ~135M parameter model. 

## HuggingFace Space

[SmolLM2 GPT Text Generator Hugging Face Space App](https://huggingface.co/spaces/sagargurujula/SmolLM2-Text-Generator)

## Features

- **User-friendly interface**: A clean and responsive web page to input text and display results.
- **Real-time generation**: Text is processed and displayed almost instantly.

## Files in the Project

### `index.html`

The front-end of the application, providing a user interface for input and output.

- Includes a text area for input and a submit button to initiate text generation.
- Displays the original input and the generated continuation in separate sections.
- Features CSS styling for a clean look and responsive design.

### `model.py`

The backend model that handles text generation.

- Uses a SmolLM2 model for generating text.
- Processes the input text and generates a continuation.
- Model is based on the SmolLM2 architecture. It is a ~135M parameter model. 
- Model is trained on the Cosmopedia V2 dataset in Hugging Face.
- **Model details**: 
   ``` 
   GPT(
   (token_embedding): Embedding(49152, 576)
   (layers): ModuleList(
      (0-29): 30 x LlamaDecoderLayer(
         (self_attn): CausalSelfAttention(
         (cq_attn): Linear(in_features=576, out_features=576, bias=False)
         (ckv_attn): Linear(in_features=576, out_features=384, bias=False)
         (c_proj): Linear(in_features=576, out_features=576, bias=False)
         (rope): RotaryPositionalEmbeddings()
         )
         (input_layernorm): RMSNorm((576,), eps=1e-05, elementwise_affine=True)
         (post_attention_layernorm): RMSNorm((576,), eps=1e-05, elementwise_affine=True)
         (mlp): LlamaMLP(
         (gate_proj): Linear(in_features=576, out_features=1536, bias=False)
         (up_proj): Linear(in_features=576, out_features=1536, bias=False)
         (down_proj): Linear(in_features=1536, out_features=576, bias=False)
         (act_fn): SiLU()
         )
      )
   )
   (final_norm): RMSNorm((576,), eps=1e-05, elementwise_affine=True)
   (lm_head): Linear(in_features=576, out_features=49152, bias=False)
   )
   ```
- **Model Summary**:
   ``` 
   =========================================================================================================
   Layer (type:depth-idx)                                  Output Shape              Param #
   =========================================================================================================
   GPT                                                     [1, 2048, 49152]          --
   ├─Embedding: 1-1                                        [1, 2048, 576]            28,311,552
   ├─ModuleList: 1-2                                       --                        --
   │    └─LlamaDecoderLayer: 2-1                           [1, 2048, 576]            --
   │    │    └─RMSNorm: 3-1                                [1, 2048, 576]            576
   │    │    └─CausalSelfAttention: 3-2                    [1, 2048, 576]            884,736
   │    │    └─RMSNorm: 3-3                                [1, 2048, 576]            576
   │    │    └─LlamaMLP: 3-4                               [1, 2048, 576]            2,654,208
   │    └─LlamaDecoderLayer: 2-2                           [1, 2048, 576]            --
   │    │    └─RMSNorm: 3-5                                [1, 2048, 576]            576
   │    │    └─CausalSelfAttention: 3-6                    [1, 2048, 576]            884,736
   │    │    └─RMSNorm: 3-7                                [1, 2048, 576]            576
   │    │    └─LlamaMLP: 3-8                               [1, 2048, 576]            2,654,208
   │    └─LlamaDecoderLayer: 2-3                           [1, 2048, 576]            --
   │    │    └─RMSNorm: 3-9                                [1, 2048, 576]            576
   │    │    └─CausalSelfAttention: 3-10                   [1, 2048, 576]            884,736
   │    │    └─RMSNorm: 3-11                               [1, 2048, 576]            576
   │    │    └─LlamaMLP: 3-12                              [1, 2048, 576]            2,654,208
   │    └─LlamaDecoderLayer: 2-4                           [1, 2048, 576]            --
   │    │    └─RMSNorm: 3-13                               [1, 2048, 576]            576
   │    │    └─CausalSelfAttention: 3-14                   [1, 2048, 576]            884,736
   │    │    └─RMSNorm: 3-15                               [1, 2048, 576]            576
   │    │    └─LlamaMLP: 3-16                              [1, 2048, 576]            2,654,208
   │    └─LlamaDecoderLayer: 2-5                           [1, 2048, 576]            --
   │    │    └─RMSNorm: 3-17                               [1, 2048, 576]            576
   │    │    └─CausalSelfAttention: 3-18                   [1, 2048, 576]            884,736
   │    │    └─RMSNorm: 3-19                               [1, 2048, 576]            576
   │    │    └─LlamaMLP: 3-20                              [1, 2048, 576]            2,654,208
   │    └─LlamaDecoderLayer: 2-6                           [1, 2048, 576]            --
   │    │    └─RMSNorm: 3-21                               [1, 2048, 576]            576
   │    │    └─CausalSelfAttention: 3-22                   [1, 2048, 576]            884,736
   │    │    └─RMSNorm: 3-23                               [1, 2048, 576]            576
   │    │    └─LlamaMLP: 3-24                              [1, 2048, 576]            2,654,208
   │    └─LlamaDecoderLayer: 2-7                           [1, 2048, 576]            --
   │    │    └─RMSNorm: 3-25                               [1, 2048, 576]            576
   │    │    └─CausalSelfAttention: 3-26                   [1, 2048, 576]            884,736
   │    │    └─RMSNorm: 3-27                               [1, 2048, 576]            576
   │    │    └─LlamaMLP: 3-28                              [1, 2048, 576]            2,654,208
   │    └─LlamaDecoderLayer: 2-8                           [1, 2048, 576]            --
   │    │    └─RMSNorm: 3-29                               [1, 2048, 576]            576
   │    │    └─CausalSelfAttention: 3-30                   [1, 2048, 576]            884,736
   │    │    └─RMSNorm: 3-31                               [1, 2048, 576]            576
   │    │    └─LlamaMLP: 3-32                              [1, 2048, 576]            2,654,208
   │    └─LlamaDecoderLayer: 2-9                           [1, 2048, 576]            --
   │    │    └─RMSNorm: 3-33                               [1, 2048, 576]            576
   │    │    └─CausalSelfAttention: 3-34                   [1, 2048, 576]            884,736
   │    │    └─RMSNorm: 3-35                               [1, 2048, 576]            576
   │    │    └─LlamaMLP: 3-36                              [1, 2048, 576]            2,654,208
   │    └─LlamaDecoderLayer: 2-10                          [1, 2048, 576]            --
   │    │    └─RMSNorm: 3-37                               [1, 2048, 576]            576
   │    │    └─CausalSelfAttention: 3-38                   [1, 2048, 576]            884,736
   │    │    └─RMSNorm: 3-39                               [1, 2048, 576]            576
   │    │    └─LlamaMLP: 3-40                              [1, 2048, 576]            2,654,208
   │    └─LlamaDecoderLayer: 2-11                          [1, 2048, 576]            --
   │    │    └─RMSNorm: 3-41                               [1, 2048, 576]            576
   │    │    └─CausalSelfAttention: 3-42                   [1, 2048, 576]            884,736
   │    │    └─RMSNorm: 3-43                               [1, 2048, 576]            576
   │    │    └─LlamaMLP: 3-44                              [1, 2048, 576]            2,654,208
   │    └─LlamaDecoderLayer: 2-12                          [1, 2048, 576]            --
   │    │    └─RMSNorm: 3-45                               [1, 2048, 576]            576
   │    │    └─CausalSelfAttention: 3-46                   [1, 2048, 576]            884,736
   │    │    └─RMSNorm: 3-47                               [1, 2048, 576]            576
   │    │    └─LlamaMLP: 3-48                              [1, 2048, 576]            2,654,208
   │    └─LlamaDecoderLayer: 2-13                          [1, 2048, 576]            --
   │    │    └─RMSNorm: 3-49                               [1, 2048, 576]            576
   │    │    └─CausalSelfAttention: 3-50                   [1, 2048, 576]            884,736
   │    │    └─RMSNorm: 3-51                               [1, 2048, 576]            576
   │    │    └─LlamaMLP: 3-52                              [1, 2048, 576]            2,654,208
   │    └─LlamaDecoderLayer: 2-14                          [1, 2048, 576]            --
   │    │    └─RMSNorm: 3-53                               [1, 2048, 576]            576
   │    │    └─CausalSelfAttention: 3-54                   [1, 2048, 576]            884,736
   │    │    └─RMSNorm: 3-55                               [1, 2048, 576]            576
   │    │    └─LlamaMLP: 3-56                              [1, 2048, 576]            2,654,208
   │    └─LlamaDecoderLayer: 2-15                          [1, 2048, 576]            --
   │    │    └─RMSNorm: 3-57                               [1, 2048, 576]            576
   │    │    └─CausalSelfAttention: 3-58                   [1, 2048, 576]            884,736
   │    │    └─RMSNorm: 3-59                               [1, 2048, 576]            576
   │    │    └─LlamaMLP: 3-60                              [1, 2048, 576]            2,654,208
   │    └─LlamaDecoderLayer: 2-16                          [1, 2048, 576]            --
   │    │    └─RMSNorm: 3-61                               [1, 2048, 576]            576
   │    │    └─CausalSelfAttention: 3-62                   [1, 2048, 576]            884,736
   │    │    └─RMSNorm: 3-63                               [1, 2048, 576]            576
   │    │    └─LlamaMLP: 3-64                              [1, 2048, 576]            2,654,208
   │    └─LlamaDecoderLayer: 2-17                          [1, 2048, 576]            --
   │    │    └─RMSNorm: 3-65                               [1, 2048, 576]            576
   │    │    └─CausalSelfAttention: 3-66                   [1, 2048, 576]            884,736
   │    │    └─RMSNorm: 3-67                               [1, 2048, 576]            576
   │    │    └─LlamaMLP: 3-68                              [1, 2048, 576]            2,654,208
   │    └─LlamaDecoderLayer: 2-18                          [1, 2048, 576]            --
   │    │    └─RMSNorm: 3-69                               [1, 2048, 576]            576
   │    │    └─CausalSelfAttention: 3-70                   [1, 2048, 576]            884,736
   │    │    └─RMSNorm: 3-71                               [1, 2048, 576]            576
   │    │    └─LlamaMLP: 3-72                              [1, 2048, 576]            2,654,208
   │    └─LlamaDecoderLayer: 2-19                          [1, 2048, 576]            --
   │    │    └─RMSNorm: 3-73                               [1, 2048, 576]            576
   │    │    └─CausalSelfAttention: 3-74                   [1, 2048, 576]            884,736
   │    │    └─RMSNorm: 3-75                               [1, 2048, 576]            576
   │    │    └─LlamaMLP: 3-76                              [1, 2048, 576]            2,654,208
   │    └─LlamaDecoderLayer: 2-20                          [1, 2048, 576]            --
   │    │    └─RMSNorm: 3-77                               [1, 2048, 576]            576
   │    │    └─CausalSelfAttention: 3-78                   [1, 2048, 576]            884,736
   │    │    └─RMSNorm: 3-79                               [1, 2048, 576]            576
   │    │    └─LlamaMLP: 3-80                              [1, 2048, 576]            2,654,208
   │    └─LlamaDecoderLayer: 2-21                          [1, 2048, 576]            --
   │    │    └─RMSNorm: 3-81                               [1, 2048, 576]            576
   │    │    └─CausalSelfAttention: 3-82                   [1, 2048, 576]            884,736
   │    │    └─RMSNorm: 3-83                               [1, 2048, 576]            576
   │    │    └─LlamaMLP: 3-84                              [1, 2048, 576]            2,654,208
   │    └─LlamaDecoderLayer: 2-22                          [1, 2048, 576]            --
   │    │    └─RMSNorm: 3-85                               [1, 2048, 576]            576
   │    │    └─CausalSelfAttention: 3-86                   [1, 2048, 576]            884,736
   │    │    └─RMSNorm: 3-87                               [1, 2048, 576]            576
   │    │    └─LlamaMLP: 3-88                              [1, 2048, 576]            2,654,208
   │    └─LlamaDecoderLayer: 2-23                          [1, 2048, 576]            --
   │    │    └─RMSNorm: 3-89                               [1, 2048, 576]            576
   │    │    └─CausalSelfAttention: 3-90                   [1, 2048, 576]            884,736
   │    │    └─RMSNorm: 3-91                               [1, 2048, 576]            576
   │    │    └─LlamaMLP: 3-92                              [1, 2048, 576]            2,654,208
   │    └─LlamaDecoderLayer: 2-24                          [1, 2048, 576]            --
   │    │    └─RMSNorm: 3-93                               [1, 2048, 576]            576
   │    │    └─CausalSelfAttention: 3-94                   [1, 2048, 576]            884,736
   │    │    └─RMSNorm: 3-95                               [1, 2048, 576]            576
   │    │    └─LlamaMLP: 3-96                              [1, 2048, 576]            2,654,208
   │    └─LlamaDecoderLayer: 2-25                          [1, 2048, 576]            --
   │    │    └─RMSNorm: 3-97                               [1, 2048, 576]            576
   │    │    └─CausalSelfAttention: 3-98                   [1, 2048, 576]            884,736
   │    │    └─RMSNorm: 3-99                               [1, 2048, 576]            576
   │    │    └─LlamaMLP: 3-100                             [1, 2048, 576]            2,654,208
   │    └─LlamaDecoderLayer: 2-26                          [1, 2048, 576]            --
   │    │    └─RMSNorm: 3-101                              [1, 2048, 576]            576
   │    │    └─CausalSelfAttention: 3-102                  [1, 2048, 576]            884,736
   │    │    └─RMSNorm: 3-103                              [1, 2048, 576]            576
   │    │    └─LlamaMLP: 3-104                             [1, 2048, 576]            2,654,208
   │    └─LlamaDecoderLayer: 2-27                          [1, 2048, 576]            --
   │    │    └─RMSNorm: 3-105                              [1, 2048, 576]            576
   │    │    └─CausalSelfAttention: 3-106                  [1, 2048, 576]            884,736
   │    │    └─RMSNorm: 3-107                              [1, 2048, 576]            576
   │    │    └─LlamaMLP: 3-108                             [1, 2048, 576]            2,654,208
   │    └─LlamaDecoderLayer: 2-28                          [1, 2048, 576]            --
   │    │    └─RMSNorm: 3-109                              [1, 2048, 576]            576
   │    │    └─CausalSelfAttention: 3-110                  [1, 2048, 576]            884,736
   │    │    └─RMSNorm: 3-111                              [1, 2048, 576]            576
   │    │    └─LlamaMLP: 3-112                             [1, 2048, 576]            2,654,208
   │    └─LlamaDecoderLayer: 2-29                          [1, 2048, 576]            --
   │    │    └─RMSNorm: 3-113                              [1, 2048, 576]            576
   │    │    └─CausalSelfAttention: 3-114                  [1, 2048, 576]            884,736
   │    │    └─RMSNorm: 3-115                              [1, 2048, 576]            576
   │    │    └─LlamaMLP: 3-116                             [1, 2048, 576]            2,654,208
   │    └─LlamaDecoderLayer: 2-30                          [1, 2048, 576]            --
   │    │    └─RMSNorm: 3-117                              [1, 2048, 576]            576
   │    │    └─CausalSelfAttention: 3-118                  [1, 2048, 576]            884,736
   │    │    └─RMSNorm: 3-119                              [1, 2048, 576]            576
   │    │    └─LlamaMLP: 3-120                             [1, 2048, 576]            2,654,208
   ├─RMSNorm: 1-3                                          [1, 2048, 576]            576
   ├─Linear: 1-4                                           [1, 2048, 49152]          28,311,552
   =========================================================================================================
   Total params: 162,826,560
   Trainable params: 162,826,560
   Non-trainable params: 0
   Total mult-adds (M): 162.83
   =========================================================================================================
   Input size (MB): 0.02
   Forward/backward pass size (MB): 3938.45
   Params size (MB): 651.31
   Estimated Total Size (MB): 4589.77
   =========================================================================================================
   ```
### `app.py`

The Flask-based server that bridges the front-end and back-end.

- Defines API endpoints for interacting with the model (`/generate/`).
- Handles POST requests from the front-end, passes input to the model, and returns generated text.
- Includes error handling to manage invalid inputs or server issues.

## Prerequisites

- **Python**: Ensure Python 3.7 or higher is installed.
- **Flask**: Install Flask using `pip install flask`.
- **Model dependencies**: Install necessary packages for running the GPT model (`transformers`, `torch`, etc., if applicable).

Note: Refer `requirements.txt` file for dependencies.

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder> # if you are in the root folder
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the server:
   ```bash
   python app.py
   ```

4. Open your browser and navigate to `http://localhost:8080` to use the application.

## Logs
- [Complete Training Logs ](training_log.txt)
``` Logs
2025-02-01 01:45:41,986 - INFO - Total Parameters: 134,515,008
2025-02-01 01:45:41,986 - INFO - Trainable Parameters: 134,515,008
2025-02-01 01:45:58,592 - INFO - Epoch 0, Step 0, Loss: 11.355476, Best Loss: 11.355476
2025-02-01 01:46:09,209 - INFO - Epoch 0, Step 1, Loss: 11.328232, Best Loss: 11.328232
2025-02-01 01:46:19,635 - INFO - Epoch 0, Step 2, Loss: 11.316939, Best Loss: 11.316939
2025-02-01 01:46:27,642 - INFO - Epoch 0, Step 3, Loss: 11.357376, Best Loss: 11.316939
2025-02-01 01:46:38,116 - INFO - Epoch 0, Step 4, Loss: 11.305080, Best Loss: 11.305080
2025-02-01 01:46:46,129 - INFO - Epoch 0, Step 5, Loss: 11.334230, Best Loss: 11.305080
2025-02-01 01:46:54,134 - INFO - Epoch 0, Step 6, Loss: 11.323222, Best Loss: 11.305080
2025-02-01 01:47:02,250 - INFO - Epoch 0, Step 7, Loss: 11.343060, Best Loss: 11.305080
2025-02-01 01:47:15,049 - INFO - Epoch 0, Step 8, Loss: 10.165360, Best Loss: 10.165360
2025-02-01 01:47:23,077 - INFO - Epoch 0, Step 9, Loss: 10.364421, Best Loss: 10.165360
.
.
.
.
2025-02-01 14:15:42,483 - INFO - Epoch 0, Step 4995, Loss: 4.230450, Best Loss: 4.102502
2025-02-01 14:15:50,508 - INFO - Epoch 0, Step 4996, Loss: 4.634490, Best Loss: 4.102502
2025-02-01 14:15:58,539 - INFO - Epoch 0, Step 4997, Loss: 4.260272, Best Loss: 4.102502
2025-02-01 14:16:06,572 - INFO - Epoch 0, Step 4998, Loss: 4.510636, Best Loss: 4.102502
2025-02-01 14:16:14,623 - INFO - Epoch 0, Step 4999, Loss: 4.590177, Best Loss: 4.102502
2025-02-01 14:16:16,566 - INFO - Step 5000 Prompt: Once upon a time 
 Generated Token: [28, 281, 253, 28249, 1666, 1217, 2591, 28, 665, 436, 253, 1528, 282, 701, 617, 4161, 281, 253, 1528, 282, 701, 617, 4161, 281, 253, 1528, 282, 701, 617, 4161, 281, 253, 1528, 282, 701, 617, 4161, 281, 253, 1528, 282, 701, 617, 4161, 281, 253, 1528, 282, 701, 30] 
 Prediction: , in a faraway land called Earth, there was a group of people who lived in a group of people who lived in a group of people who lived in a group of people who lived in a group of people who lived in a group of people.
2025-02-01 14:16:24,595 - INFO - Epoch 0, Step 5000, Loss: 4.321911, Best Loss: 4.102502
2025-02-01 14:16:27,485 - INFO - Max iterations reached. Training stopped.
2025-02-01 14:16:27,506 - INFO - Training completed!
2025-02-01 14:16:27,515 - INFO - Final Loss: 4.321911
2025-02-01 14:16:27,515 - INFO - Best Loss Achieved: 4.102502
2025-02-01 14:16:27,515 - INFO - Best Model Saved To: /kaggle/working/best_model.pth
2025-02-01 14:16:27,515 - INFO - Checpoint Model Saved To: /kaggle/working/checkpoint_model.pth
2025-02-01 14:39:27,845 - INFO - Resuming from epoch 0, step 5001, best loss 4.102502
2025-02-01 14:39:27,847 - INFO - Total Parameters: 134,515,008
2025-02-01 14:39:27,847 - INFO - Trainable Parameters: 134,515,008
2025-02-01 14:39:48,115 - INFO - Epoch 0, Step 5001, Loss: 4.682718, Best Loss: 4.102502
2025-02-01 14:39:56,142 - INFO - Epoch 0, Step 5002, Loss: 4.552681, Best Loss: 4.102502
2025-02-01 14:40:04,168 - INFO - Epoch 0, Step 5003, Loss: 4.262938, Best Loss: 4.102502
2025-02-01 14:40:12,198 - INFO - Epoch 0, Step 5004, Loss: 4.162656, Best Loss: 4.102502
2025-02-01 14:40:20,229 - INFO - Epoch 0, Step 5005, Loss: 4.282134, Best Loss: 4.102502
```


## Hugging Face App Screenshots

![Example 1](Hugging%20Face%20Space%20Example%201.png)

![Example 2](Hugging%20Face%20Space%20Example%202.png)

