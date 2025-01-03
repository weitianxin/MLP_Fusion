OT的具体都在switch-OT.ipynb这个notebook里面
所有改模型的方法都在

```
def extract_by_average(
  MLP: SwitchTransformersSparseMLP,
  sparsity,
  extract_type = "average",
  max_it = 150,
  wd_extract = None,
  T = None  
):
```

这个函数里面

evaluate都是load模型，把模型改完之后再算prediction，这一块在notebook最后。loss就在这个的上面。