训练参数为rank=32， alpha = 64，batch_size = 8, max_steps = 20, torch.bfloat16 客户端20选2 200轮

任务1：训练commonsense170k，测试数据集"boolq", "piqa", "social_i_qa", "ARC-Challenge", "ARC-Easy", "openbookqa", "hellaswag", "winogrande"
fedit
{'ARC-Challenge': 0.6339590443686007, 'ARC-Easy': 0.8194444444444444, 'boolq': 0.6577981651376147, 'piqa': 0.7910772578890098, 'social_i_qa': 0.7466734902763562, 'openbookqa': 0.698, 'hellaswag': 0.8812985461063533, 'winogrande': 0.7758484609313339}

ffalora
{'ARC-Challenge': 0.6160409556313993, 'ARC-Easy': 0.7954545454545454, 'boolq': 0.6510703363914373, 'piqa': 0.7883569096844396, 'social_i_qa': 0.7118730808597749, 'openbookqa': 0.65, 'hellaswag': 0.8388767177853017, 'winogrande': 0.7348066298342542}

flora
{'ARC-Challenge': 0.4872013651877133, 'ARC-Easy': 0.6948653198653199, 'boolq': 0.6302752293577981, 'piqa': 0.7404787812840044, 'social_i_qa': 0.6074718526100307, 'openbookqa': 0.512, 'hellaswag': 0.6614220274845648, 'winogrande': 0.5146014206787688}

fedls 1.17-0.77
```python
mask = sample_layers(l2_norm, entropy, rho=1, min_prob=0.4) max_prob=0.99

with torch.no_grad():
    if len(buffer.keys()) > 0:
        for key in global_dict.keys():
            global_dict[key] = avg_dict[key] + 0.05 * buffer[key]
            buffer[key] = 0.5 * buffer[key] + 0.5 * delta_dict[key]
    else:
        for key in global_dict.keys():
            global_dict[key] = avg_dict[key]
            buffer[key] = delta_dict[key]
```

{'ARC-Challenge': 0.6390784982935154, 'ARC-Easy': 0.7975589225589226, 'boolq': 0.6519877675840978, 'piqa': 0.8079434167573449, 'social_i_qa': 0.7415557830092119, 'openbookqa': 0.666, 'hellaswag': 0.8682533359888468, 'winogrande': 0.7355958958168903}
```python
        min_prob = 0.4 rho = 1
        mask_dict = {}
        for i in range(len(mask)):
            mask_dict[names[i]] = mask[i]
        with torch.no_grad():
            if len(buffer.keys()) > 0:
                for key in global_dict.keys():
                    if mask_dict[key] == 0:
                        global_dict[key] = avg_dict[key] + 0.1 * buffer[key]
                        buffer[key] = 0.8 * buffer[key] + 0.2 * delta_dict[key]
                    else:
                        global_dict[key] = avg_dict[key]
            else:
                for key in global_dict.keys():
                    global_dict[key] = avg_dict[key]
                    buffer[key] = delta_dict[key]
```
1.17 - 0.7
{'ARC-Challenge': 0.6348122866894198, 'ARC-Easy': 0.8013468013468014, 'boolq': 0.6568807339449542, 'piqa': 0.7976060935799782, 'social_i_qa': 0.736949846468782, 'openbookqa': 0.672, 'hellaswag': 0.8643696474805815, 'winogrande': 0.7277032359905288}

```python
        mask = sample_layers(l2_norm, entropy, rho=0.5, min_prob=0.4, max_prob=0.9)
        mask_dict = {}
        for i in range(len(mask)):
            mask_dict[names[i]] = mask[i]
        with torch.no_grad():
            if len(buffer.keys()) > 0:
                for key in global_dict.keys():
                    # if mask_dict[key] == 0:
                    #     global_dict[key] = avg_dict[key] + 0.5 * buffer[key]
                    #     # buffer[key] = 0.8 * buffer[key] + 0.2 * delta_dict[key]
                    # else:
                    global_dict[key] = avg_dict[key]
            else:
                for key in global_dict.keys():
                    global_dict[key] = avg_dict[key]
                    buffer[key] = delta_dict[key]
        return mask, names
```
1.17 - 0.85
{'ARC-Challenge': 0.6313993174061433, 'ARC-Easy': 0.805976430976431, 'boolq': 0.655045871559633, 'piqa': 0.795429815016322, 'social_i_qa': 0.7446264073694985, 'openbookqa': 0.694, 'hellaswag': 0.8733320055765784, 'winogrande': 0.7434885556432518}
mmlu 0.2161
tunr 1 avg: 5.98
turn 2 avg: 5.40
avg: 5.69

rho = 0.9
{'ARC-Challenge': 0.6305460750853242, 'ARC-Easy': 0.7988215488215489, 'boolq': 0.653211009174312, 'piqa': 0.8025027203482046, 'social_i_qa': 0.7502558853633572, 'openbookqa': 0.682, 'hellaswag': 0.8621788488348935, 'winogrande': 0.7348066298342542}
mmlu 0.2221
tunr 1 avg: 6.04
turn 2 avg: 4.97
avg: 5.51

rho = 1
1.17 -> 0.7
{'ARC-Challenge': 0.6254266211604096, 'ARC-Easy': 0.8013468013468014, 'boolq': 0.6529051987767585, 'piqa': 0.8025027203482046, 'social_i_qa': 0.7410440122824974, 'openbookqa': 0.67, 'hellaswag': 0.8678550089623581, 'winogrande': 0.7434885556432518}
mmlu 0.2174
tunr 1 avg: 5.80
turn 2 avg: 4.84
avg: 5.32


{'ARC-Challenge': 0.6203071672354948, 'ARC-Easy': 0.8042929292929293, 'boolq': 0.6587155963302752, 'piqa': 0.7965179542981502, 'social_i_qa': 0.7461617195496417, 'openbookqa': 0.684, 'hellaswag': 0.8676558454491137, 'winogrande': 0.7403314917127072}
0.2171



任务2：训练Alpaca GPT-4，测试数据集MMLU test以及MT-Bench
测试结果是比较稳定，测试一次就可以了
| method  | MMLU   |
| ---     | ---    |
| FedIT   | 0.2233 |
| FFALora | 0.2065 |
| FLora   | 0.2301 |
| FedLS   | 0.2104 |
| FedLS (0.1) | 0.1929 |

测试结果是比较稳定，测试一次就可以了
| method  | MT-1 | MT-2 | AVG  |
| ---     | ---  | ---  | ---  |
| FedIT   | 6.20 | 4.33 | 5.26 |
| FFALora | 6.28 | 4.49 | 5.38 |
| FLora   | 6.00 | 4.51 | 5.26 |
| FedLS   | 6.12 | 4.51 | 5.32 |
| FedLS (0.1)   | 5.97 | 4.45 | 5.21 |

任务3：在gsm8k上训练然后测试  
训练参数改为200轮
训练结果很差，在联邦场景下，数学推理很差 可工作的地方
| method  | gsm8k   |
| ---     | ---    |
| FedIT   |  |
| FFALora |  |
| FLora   |  |
| FedLS   |  |