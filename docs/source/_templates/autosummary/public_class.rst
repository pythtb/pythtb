{{ fullname | escape | underline }}

.. autoclass:: {{ fullname }}

{% set exclude_special = ["__init__"] %}

{% if methods %}
Methods
-------
.. autosummary::
   :nosignatures:
   :toctree: {{ fullname | replace(".", "/") }}/

   {% for item in methods if (not item.startswith("_")) and (item not in exclude_special) -%}
   {{ fullname }}.{{ item }}
   {% endfor %}
{% endif %}

{% if attributes %}
Attributes
----------
.. autosummary::
   :nosignatures:
   :toctree: {{ fullname | replace(".", "/") }}/

   {% for item in attributes if (not item.startswith("_")) -%}
   {{ fullname }}.{{ item }}
   {% endfor %}
{% endif %}