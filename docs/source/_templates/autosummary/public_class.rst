{{ fullname | escape | underline }}

.. autoclass:: {{ fullname }}

{% set exclude_special = ["__init__"] %}

{% set m = namespace(items=[]) %}
{% for item in methods -%}
{% set name = item.strip() %}
{% if name and (name not in exclude_special) and (not name.startswith('_')) and (not (name.startswith('__') and name.endswith('__'))) %}
{% set m.items = m.items + [name] %}
{% endif %}
{% endfor %}

{% if m.items %}
Methods
-------
.. autosummary::
   :nosignatures:
   :toctree: {{ fullname | replace(".", "/") }}/

   {% for name in m.items -%}
   {{ fullname }}.{{ name }}
   {% endfor %}
{% endif %}

{% set a = namespace(items=[]) %}
{% for item in attributes -%}
{% set name = item.strip() %}
{% if name and (not name.startswith('_')) and (not (name.startswith('__') and name.endswith('__'))) %}
{% set a.items = a.items + [name] %}
{% endif %}
{% endfor %}

{% if a.items %}
Attributes
----------
.. autosummary::
   :nosignatures:
   :toctree: {{ fullname | replace(".", "/") }}/

   {% for name in a.items -%}
   {{ fullname }}.{{ name }}
   {% endfor %}
{% endif %}
