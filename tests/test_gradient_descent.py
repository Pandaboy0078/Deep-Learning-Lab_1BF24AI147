import numpy as np
import pytest
from lab01.gradient_descent import gradient_descent,make_regression_data

def test_all_methods_reduce_loss():
    x,y=make_regression_data(n=120)
    for batch in (120,1,16):
        result=gradient_descent(x,y,batch_size=batch,epochs=60)
        assert np.isfinite(result["weight"])
        assert abs(result["weight"]-3)<.4
        assert abs(result["bias"]-2)<.3
        assert result["epoch_losses"][-1]<result["step_losses"][0][1]

def test_invalid_batch():
    x,y=make_regression_data()
    with pytest.raises(ValueError):
        gradient_descent(x,y,batch_size=0)
