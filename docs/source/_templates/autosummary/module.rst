{{ fullname | escape | underline }}

-----------
Description
-----------

.. automodule:: {{ fullname }}

.. currentmodule:: {{ fullname }}

{% if classes %}

-------
Classes
-------

.. autosummary::
    :toctree: .
    {% for class in classes %}

    {{ class }}

    {% endfor %}

{% endif %}

{% if functions %}

---------
Functions
---------

.. autosummary::
    :toctree: .
    {% for function in functions %}
    {{ function }}
    {% endfor %}

{% endif %}