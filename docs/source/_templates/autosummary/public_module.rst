{{ fullname }}
{{ underline }}

{# PACKAGE PAGE (e.g. pythtb.io): show ONLY submodules, no members #}
{% if modules and (not functions and not classes and not attributes and not exceptions) %}
.. automodule:: {{ fullname }}
   :no-members:

Submodules
----------
.. autosummary::
   :toctree: .
   :nosignatures:
   :template: autosummary/public_class.rst
{% for m in modules %}
   {{ m }}
{% endfor %}

{% else %}

{# SUBMODULE PAGE (e.g. pythtb.io.wannier90): show docstring, then public members.
   Also generate stub pages so items are clickable and appear in the sidebar. #}
{% set module_doc = obj.__doc__ if obj and obj.__doc__ else None %}
{% if module_doc %}
{{ module_doc }}

{% endif %}
.. automodule:: {{ fullname }}
   :no-members:

.. currentmodule:: {{ fullname }}

{% if classes %}
Classes
-------
.. autosummary::
   :toctree: .
   :nosignatures:
   :template: autosummary/public_class.rst
   
{% for c in classes %}
{% if not c.startswith('_') %}
   {{ c }}
{% endif %}
{% endfor %}
{% endif %}

{% if functions %}
Functions
---------
.. autosummary::
   :toctree: .
   :nosignatures:
{% for f in functions %}
{% if not f.startswith('_') %}
   {{ f }}
{% endif %}
{% endfor %}
{% endif %}

{% if attributes %}
Data
----
.. autosummary::
   :toctree: .
{% for a in attributes %}
{% if not a.startswith('_') %}
   {{ a }}
{% endif %}
{% endfor %}
{% endif %}

{% if exceptions %}
Exceptions
----------
.. autosummary::
   :toctree: .
   :nosignatures:
{% for e in exceptions %}
{% if not e.startswith('_') %}
   {{ e }}
{% endif %}
{% endfor %}
{% endif %}

{% endif %}
