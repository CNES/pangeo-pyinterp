{{ fullname }}
{{ underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:

   {% block methods %}

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
            item not in ['__annotations__',
                         '__dict__',
                         '__doc__',
                         '__entries',
                         '__getstate__',
                         '__hash__',
                         '__init__',
                         '__members__',
                         '__module__',
                         '__setstate__',
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
