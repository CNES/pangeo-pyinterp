{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:
   {% block methods %}

   {%- set attr = [] -%}
   {%- set meth = [] -%}
   {%- set private = [] -%}
   {%- set protected = [] -%}
   {%- set special = [] -%}
   {%- set inherited_meth = [] -%}
   {%- set skip = ['__abstractmethods__',
                   '__annotations__',
                   '__dict__',
                   '__doc__',
                   '__entries',
                   '__hash__',
                   '__init__',
                   '__members__',
                   '__module__',
                   '__slots__',
                   '__weakref__'] -%}

   {%- for item in methods if not item in skip -%}
      {%- if item in inherited_members -%}
         {{ inherited_meth.append(item) or "" }}
      {%- else -%}
         {{ meth.append(item) or "" }}
      {%- endif -%}
   {%- endfor -%}

   {%- for item in members
         if not item in inherited_members and not item in skip -%}
      {%- if item.startswith('__') and item.endswith('__') -%}
         {{ special.append(item) or "" }}
      {%- elif item.startswith('__') -%}
         {{ private.append(item) or "" }}
      {%- elif item.startswith('_') -%}
         {{ protected.append(item) or "" }}
      {%- endif -%}
   {%- endfor %}

   {%- if fullname == 'pyinterp.Swath' %}
      {# autodoc doesn't handle inherited attributes of dataclasses. #}
      {%- for item in ['equator_coordinates',
                       'lon_nadir',
                       'lat_nadir',
                       'time',
                       'x_al'] %}
         {{ attributes.remove(item) or "" }}
      {%- endfor -%}
   {%- endif %}

   {%- if attributes %}
   .. rubric:: {{ _('Attributes') }}
   .. autosummary::
      :toctree:
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif -%}

   {%- if meth %}
   .. rubric:: {{ _('Public Methods') }}
   .. autosummary::
      :toctree:
   {% for item in meth %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif -%}

   {%- if protected %}
   .. rubric:: {{ _('Protected Methods') }}
   .. autosummary::
      :toctree:
   {% for item in protected %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif -%}

   {%- if private %}
   .. rubric:: {{ _('Private Methods') }}
   .. autosummary::
      :toctree:
   {% for item in private %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif -%}

   {%- if special %}

   .. rubric:: {{ _('Special Methods') }}
   .. autosummary::
      :toctree:
   {% for item in special %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {%- endif -%}

   {%- if inherited_meth %}

   .. rubric:: {{ _('Inherited Methods') }}
   .. autosummary::
      :toctree:
   {% for item in inherited_meth %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {%- endif -%}

   {%- endblock -%}
