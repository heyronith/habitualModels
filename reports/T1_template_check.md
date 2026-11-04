# T1 template check

Primary live check on **Qwen3-1.7B** (thinking disabled). Tiny CPU model check retained in `T1_template_check_tiny.md`.

model_key: `qwen3_1_7b`
revision: `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`

## Rendered string up to answer position

```
<|im_start|>system
You manage the supply desk of a small workshop. For each order you send a cart to one dock. Reply with the dock number only.<|im_end|>
<|im_start|>user
Order #167 for Ava: 1 spoft.
Recipe: 3 dax + 2 vexa.
Docks: Dock 5 supplies dax. Dock 9 supplies vexa.
Storeroom: dax 1/20, vexa 0/20.
Which dock should the cart go to?<|im_end|>
<|im_start|>assistant
<think>

</think>


```

length_chars: 391

## Token checks (answer-position candidates)

| text | token_ids | single |
|---|---|---|
| 2 | [17] | True |
| 3 | [18] | True |
| 4 | [19] | True |
| 5 | [20] | True |
| 6 | [21] | True |
| 7 | [22] | True |
| 8 | [23] | True |
| 9 | [24] | True |
| yes | [9693] | True |
| no | [2152] | True |

all_single: True

## Restricted vs renormalised full softmax

dock_a=5, dock_b=9
max_abs_diff: 5.960464477539063e-08
equal_atol_1e-5: True
