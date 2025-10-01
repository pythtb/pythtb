{{ fullname | escape | underline }}

.. autoclass:: {{ fullname }}
   :members:
   :undoc-members:
   :member-order: bysource
{#  omit :inherited-members: if you don't want inherited methods listed below #}

{# Hide “Bases: object” by NOT using :show-inheritance: #}

{# ---- Filtering helpers ---- #}
{% set HIDE = ["__dict__", "__weakref__", "__module__", "__annotations__", "__doc__"] %}
{% macro base(item) -%}{{ item.lstrip('~').split('.')[-1] }}{%- endmacro %}

{# ---- Public Methods (summary table only; no per-method pages) ---- #}
{% block methods %}
{% if methods %}
.. rubric:: Public Methods

.. autosummary::
   :nosignatures:
{#  NOTE: no :toctree:, so this is a table only, no stub generation #}

{% for item in methods %}
{% set b = base(item) %}
{% if not b.startswith('_') and b not in HIDE %}
   ~{{ fullname }}.{{ b }}
{% endif %}
{% endfor %}
{% endif %}
{% endblock %}

{# ---- Public Attributes (summary table only) ---- #}
{% block attributes %}
{% if attributes %}
.. rubric:: Public Attributes

.. autosummary::
   :nosignatures:

{% for item in attributes %}
{% set b = base(item) %}
{% if not b.startswith('_') and b not in HIDE %}
   ~{{ fullname }}.{{ b }}
{% endif %}
{% endfor %}
{% endif %}
{% endblock %}