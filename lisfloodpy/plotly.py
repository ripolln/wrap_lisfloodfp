#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# plotly lib


import plotly.offline as go_offline
import plotly.graph_objects as go
import plotly
from plotly.subplots import make_subplots

# # plotly colors built-in 
# viridis_cmap = plotly.colors.PLOTLY_SCALES['Viridis']
# div_cmap = plotly.colors.diverging.RdYlBu
# cmo_cmap = plotly.colors.cmocean.haline

def fig_surface(x, y, z, cmap, w, h, xmin, xmax, ymin, ymax):
    '''
    SINGLE 3D SURFACE PLOT
    
    x, y, z     axis data
    cmap        colormap 
    w, h        width, height
    xmin, xmax  axis coordinate limits
    ymin, ymax
    '''
    
    fig=go.Figure()
    
    fig.add_trace(go.Surface(z=z, x=x, y=y, 
                             colorscale=cmap,
                             
                             # adds lightning
                             lighting=dict(ambient=0.5, diffuse=1, fresnel=4, specular=0.5, roughness=0.5),
                             lightposition=dict(x=-810, y=-900, z=10000)
    ))
    
    # show contours
    fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                      highlightcolor="limegreen", project_z=True))
    
    fig.update_layout(width=w, height=h,
                      
                      # changes camera perspective
                      scene_camera_eye=dict(x=0.07, y=-1.88, z=1.64),
                      
                      # changes properties
                      scene=dict(aspectratio=dict(x=2, y=2, z=0.25),
                                 xaxis = dict(range=[xmin,xmax]),
                                 yaxis = dict(range=[ymin,ymax]),
                     ))
    
    #go_offline.plot(fig,filename='F:/3D_Terrain/3d_terrain.html',validate=True, auto_open=False)
    
    fig.show()


def fig_surface_subplots(x1, y1, z1, x2, y2, z2, cmap, h, xmin, xmax, ymin, ymax,
                         title1, title2, xlight=-810, ylight=-900, zlight=10000):
    '''
    SINGLE 3D SURFACE PLOT
    
    x, y, z     axis data
    cmap        colormap 
    w, h        width, height
    xmin, xmax  axis coordinate limits
    ymin, ymax
    '''
    
    # Initialize figure with 2 3D subplots
    fig = make_subplots(
        rows=1, cols=2, 
        column_widths=[0.5, 0.5],
        subplot_titles = (title1, title2),
        specs=[[{'type': 'scene'}, {'type': 'scene'}],] 
               #[{'is_3d': True}, {'is_3d': True}]],
               )
    
    # adding surfaces to subplots.
    fig.add_trace(go.Surface(z=z1, x=x1, y=y1, 
                             colorscale=cmap, 
                             lighting=dict(ambient=0.5, diffuse=1, fresnel=4, specular=0.5, roughness=0.5),
                             lightposition=dict(x=xlight, y=ylight, z=zlight)
                            ), row=1, col=1,
                 )
    
    fig.add_trace(go.Surface(z=z2, x=x2, y=y2, 
                             colorscale=cmap,
                             lighting=dict(ambient=0.5, diffuse=1, fresnel=4, specular=0.5, roughness=0.5),
                             lightposition=dict(x=xlight, y=ylight, z=zlight)
                            ), row=1, col=2,
                 )
    
    fig.update_traces(contours_z=dict(show=False, usecolormap=True,
                                      highlightcolor="limegreen", project_z=True))
    
    fig.update_layout(height=h,
                      scene_camera_eye=dict(x=0.07, y=-1.88, z=1.64),
                      scene2_camera_eye=dict(x=0.07, y=-1.88, z=1.64),
                      scene=dict(aspectratio=dict(x=2, y=2, z=0.25),
                                 xaxis = dict(range=[xmin,xmax]),
                                 yaxis = dict(range=[ymin,ymax]),),
                      scene2=dict(aspectratio=dict(x=2, y=2, z=0.25),
                                 xaxis = dict(range=[xmin,xmax]),
                                 yaxis = dict(range=[ymin,ymax]),)
                      )
    
    fig = go.Figure(fig)
    
    # make share camera for second subplot
    from IPython.core.display import display, HTML
    from plotly.offline import plot
    from plotly import tools
    
    div = plot(fig, include_plotlyjs=False, output_type='div')
    # retrieve the div id (you probably want to do something smarter here with beautifulsoup)
    div_id = div.split('=')[1].split()[0].replace("'", "").replace('"', '')
    # your custom JS code
    js = '''
        <script>
        var gd = document.getElementById('{div_id}');
        var isUnderRelayout = false
    
        gd.on('plotly_relayout', () => {{
          console.log('relayout', isUnderRelayout)
          if (!isUnderRelayout) {{
            Plotly.relayout(gd, 'scene2.camera', gd.layout.scene.camera)
              .then(() => {{ isUnderRelayout = false }}  )
          }}
    
          isUnderRelayout = true;
        }})
        </script>'''.format(div_id=div_id)
    # merge everything
    div = '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>' + div + js
    # show the plot 
    display(HTML(div))
    
    
def fig_surface_heatmap(z, cmap, w, h, xtitle, ytitle, ztitle, xlight=500, ylight=100, zlight=2000):    
    '''
    SINGLE 3D SURFACE / HEATMAP PLOT
    
    x, y, z     axis data
    cmap        colormap 
    w, h        width, height
    '''
    
    fig=go.Figure()
    
    fig.add_trace(go.Surface(z=z, 
                             #x=xutm,
                             #y=np.flipud(yutm), 
                             colorscale=cmap,
                             lighting=dict(ambient=0.5, diffuse=1, fresnel=4, specular=0.5, roughness=0.5),
                             lightposition=dict(x=xlight, y=ylight, z=zlight)
                            ))
    
    # Update plot sizing
    fig.update_layout(
        width=w,
        height=h,
        autosize=False,
        margin=dict(t=100, b=0, l=0, r=0),
        template="plotly_white",
    )
    
    # Update 3D scene options
    fig.update_scenes(
        aspectratio=dict(x=2, y=2, z=0.25),
        aspectmode="manual",
        xaxis_autorange="reversed",
        yaxis_autorange="reversed"
    )
    
    # update set axes title
    fig.update_layout(scene = dict(
                        xaxis_title=xtitle,
                        yaxis_title=ytitle,
                        zaxis_title=ztitle),
                        #xaxis_showticklabels=False,
                        width=w, height=h,
                        margin=dict(r=20, b=10, l=10, t=10))
    
    # Add dropdown
    fig.update_layout(
        updatemenus=[
            dict(
                type = "buttons",
                direction = "left",
                buttons=list([
                    dict(
                        args=["type", "surface"],
                        label="3D Surface",
                        method="restyle"
                    ),
                    dict(
                        args=["type", "heatmap"],
                        label="Heatmap",
                        method="restyle"
                    )
                ]),
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.11,
                xanchor="left",
                y=1.1,
                yanchor="top"
            ),
        ]
    )
    
    # Add annotation
    fig.update_layout(
        annotations=[
            dict(text="Trace type:", showarrow=False,
                                 x=0, y=1.08, yref="paper", align="left")
        ]
    )
    
    fig.show()

def fig_heatmap(x, y, z, cmap, w, h):    
    '''
    SINGLE HEATMAP PLOT (it can be georeference indicating x, y)
    
    x, y, z     axis data
    cmap        colormap 
    w, h        width, height
    '''
    
    # Create figure heatmap
    fig = go.Figure()
    
    # Add surface trace
    fig.add_trace(go.Heatmap(z=z, 
                             x=x,
                             y=y, 
                             colorscale=cmap))
    
    # Update plot sizing
    fig.update_layout(
        width=w,
        height=h,
        autosize=False,
        margin=dict(t=100, b=0, l=0, r=0),
        template="plotly_white",
    )
    
    # Update 3D scene options
    fig.update_scenes(
        aspectratio=dict(x=1, y=1, z=0.7),
        aspectmode="manual",
        xaxis_autorange="reversed",
        yaxis_autorange="reversed"
    )
    
    fig.show()