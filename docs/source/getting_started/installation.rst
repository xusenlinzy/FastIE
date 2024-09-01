.. _installation:

安装
============

FastIE是一个简单易用的信息抽取模型训练推理框架

依赖
------------

* OS: Linux或者Windows
* Python: 3.8 -- 3.12
* GPU: 推荐计算能力大于 7.0

pip安装
----------------

.. code-block:: console

    $ # (Recommended) Create a new conda environment.
    $ conda create -n myenv python=3.10 -y
    $ conda activate myenv

    $ pip install fastie

.. _build_from_source:

源码安装
-----------------

.. code-block:: console

    $ git clone https://github.com/xusenlinzy/fastie.git
    $ cd fastie
    $ pip install -e .
