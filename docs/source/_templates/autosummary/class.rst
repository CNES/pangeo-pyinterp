{{ fullname }}
{{ underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:

   {% block methods %}
   .. automethod:: __init__

   {% set meth = [] -%}

   {%- for item in methods -%}
      {%- if item != '__init__' -%}
        {{ meth.append(item) or "" }}
      {%- endif -%}
   {%- endfor -%}

   {%- for item in members -%}
      {%- if item not in inherited_members and
            item not in attributes and
            item not in meth and
            item not in ['__dict__',
                         '__entries',
                         '__doc__',
                         '__init__',
                         '__members__',
                         '__module__',
                         '__weakref__'] -%}
        {{ meth.append(item) or "" }}
      {%- endif -%}
   {%- endfor -%}

   {%- if meth -%}
   .. rubric:: Methods
   .. autosummary::
      :toctree:

   {% for item in meth %}
      {{ name }}.{{ item }}
   {%- endfor %}
   {%- endif -%}

   {%- endblock -%}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Attributes
   .. autosummary::
      :toctree:

   {% for item in attributes %}
      {{ name }}.{{ item }}
   {%- endfor %}
   {%- endif -%}
   {% endblock %}