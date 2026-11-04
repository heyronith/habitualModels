# T1 template check

model_key: `tiny`

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
| 2 | [8] | True |
| 3 | [9] | True |
| 4 | [10] | True |
| 5 | [11] | True |
| 6 | [12] | True |
| 7 | [13] | True |
| 8 | [14] | True |
| 9 | [15] | True |
| yes | [16] | True |
| no | [17] | True |

all_single: True

## Restricted vs renormalised full softmax

dock_a=5, dock_b=9
max_abs_diff: 2.9802322387695312e-08
equal_atol_1e-5: True
