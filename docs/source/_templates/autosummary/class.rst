{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   {% block attributes %}
   {% if attributes %}

----------
Attributes
----------

.. autosummary::
   :toctree:
{% for item in all_attributes %}
   {%- if not item.startswith('_') %}
   {{ name }}.{{ item }}
   {%- endif -%}
{%- endfor %}

   {% endif %}
   {% endblock %}

   {% if methods %}

----------
Methods
----------

.. autosummary::
   :toctree:
{% for item in all_methods %}
   {%- if not item.startswith('_') or item in ['__call__'] %}
   {{ name }}.{{ item }}
   {%- endif -%}
{%- endfor %}

   {% endif %}
   {% endblock %}