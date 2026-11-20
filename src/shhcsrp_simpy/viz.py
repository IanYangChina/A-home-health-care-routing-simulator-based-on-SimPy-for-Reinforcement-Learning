"""Interactive, entirely offline Plotly HTML report from simulation event logs."""
from __future__ import annotations
from collections import defaultdict
from html import escape
from pathlib import Path
import json
import plotly.graph_objects as go

from .data import Scenario


def export_html(scenario: Scenario, events: list[dict], metrics: dict,
                output: str | Path, *, title='SHHCSRP SimPy / Gymnasium demo',
                policy: str = 'unknown', total_reward: float | None = None):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    palette = ['#1d4ed8','#0e7490','#7c3aed','#b45309','#c2410c','#047857',
               '#be185d','#0f766e','#64748b','#a16207','#4f46e5','#991b1b',
               '#1e40af','#0369a1','#86198f','#115e59','#334155','#9a3412']
    depot = scenario.depot
    fig_map = go.Figure()
    fig_map.add_trace(go.Scatter(x=[depot[0]], y=[depot[1]], mode='markers+text',
        name='Depot', text=['Depot'], textposition='top center',
        marker={'size':16,'color':'#0f172a','symbol':'diamond'}))
    fig_map.add_trace(go.Scatter(
        x=[j.position[0] for j in scenario.jobs], y=[j.position[1] for j in scenario.jobs],
        mode='markers',name='Appointments',marker={'size':7,'color':'#b8c5d8'},
        text=[f'Job {j.id} / Elder {j.elder_id} / Skill {j.demand}<br>'
              f'Window {j.window_start:.1f}–{j.window_end:.1f} min'
              for j in scenario.jobs],hovertemplate='%{text}<extra></extra>'))
    routes = defaultdict(list)
    service_start = {}
    visits = defaultdict(list)
    bars = defaultdict(list)
    pending_travel = {}
    pending_wait = {}
    for ev in events:
        w,j = ev.get('nurse'),ev.get('job')
        t = float(ev['time'])
        if ev['type']=='travel_start':
            if not routes[w]:
                routes[w].append((float(ev['x0']),float(ev['y0'])))
            pending_travel[w]=(t,j)
        elif ev['type']=='arrive':
            if w in pending_travel:
                t0,j0=pending_travel.pop(w)
                bars['Travel'].append((w,t0,t,j0))
            pos=scenario.jobs[j].position
            routes[w].append((pos[0],pos[1]))
            visits[w].append(j)
        elif ev['type']=='wait_start':
            pending_wait[(w,j)]=t
        elif ev['type']=='service_start':
            if (w,j) in pending_wait:
                bars['Waiting'].append((w,pending_wait.pop((w,j)),t,j))
            service_start[(w,j)] = t
        elif ev['type']=='service_end':
            if (w,j) in service_start:
                bars['Service'].append((w,service_start.pop((w,j)),t,j))
        elif ev['type']=='return_start':
            pending_travel[w]=(t,None)
        elif ev['type']=='return_end':
            if w in pending_travel:
                t0,j0=pending_travel.pop(w)
                bars['Return'].append((w,t0,t,None))
                routes[w].append((depot[0],depot[1]))
    for w, xy in sorted(routes.items()):
        fig_map.add_trace(go.Scatter(x=[p[0] for p in xy],y=[p[1] for p in xy],
            mode='lines+markers', name=f'Nurse {w}',
            line={'color':palette[w%len(palette)],'width':2.5},
            marker={'size':8},
            customdata=[str(z) for z in ['depot']+visits[w]+(['depot'] if len(xy)>len(visits[w])+1 else [])][:len(xy)],
            hovertemplate='Nurse '+str(w)+' / Job %{customdata}<extra></extra>'))
    fig_map.update_layout(title='Routes and appointment locations (XY)',
                          xaxis_title='X', yaxis_title='Y',height=510, legend={'orientation':'h'},
                          margin={'l':40,'r':20,'b':40,'t':60}, template='plotly_white',
                          hovermode='closest')
    fig_map.update_yaxes(scaleanchor='x', scaleratio=1)
    fig_time = go.Figure()
    colors = {'Travel':'#64748b','Waiting':'#eab308','Service':'#16a34a','Return':'#2563eb'}
    for kind in ['Travel','Waiting','Service','Return']:
        records=bars[kind]
        fig_time.add_trace(go.Bar(y=[f'Nurse {w}' for w,a,b,j in records],
            x=[max(0,b-a) for w,a,b,j in records],
            base=[a for w,a,b,j in records], name=kind, orientation='h',
            marker={'color':colors[kind]},
            customdata=[str(j if j is not None else 'depot') for w,a,b,j in records],
            hovertemplate=kind+' | %{y}<br>Start: %{base:.2f} min<br>'
                          'Duration: %{x:.2f} min<br>Job: %{customdata}<extra></extra>'))
    fig_time.update_layout(title='Nurse activities along simulation time',
        barmode='overlay', xaxis_title='Simulated minutes',yaxis_title='',height=max(320,45*len(scenario.nurse_skills)),
        template='plotly_white',margin={'l':85,'r':20,'b':50,'t':65},
        legend={'orientation':'h'}, yaxis={'autorange':'reversed'})
    chart1=fig_map.to_html(full_html=False,include_plotlyjs='inline',config={'responsive':True})
    chart2=fig_time.to_html(full_html=False,include_plotlyjs=False,config={'responsive':True})
    cards = [('Jobs served',f"{metrics['jobs_completed']} / {metrics['jobs_total']}"),
             ('Completion rate',f"{100*metrics['completion_rate']:.1f}%"),
             ('Travel',f"{metrics['travel_minutes']:.1f} min"),
             ('Waiting',f"{metrics['waiting_minutes']:.1f} min"),
             ('Service',f"{metrics['service_minutes']:.1f} min"),
             ('Simulated duration',f"{metrics['time']:.1f} min")]
    if total_reward is not None:
        cards.append(('Total reward',f'{total_reward:.2f}'))
    card_html=''.join(f'<div class="metric"><div class="key">{escape(k)}</div>'
                      f'<div class="value">{escape(v)}</div></div>' for k,v in cards)
    rows='\n'.join('<tr><td>'+str(i+1)+'</td><td>'+str(ev['time'])+'</td><td>'+escape(str(ev.get('type','')))
                   +'</td><td>'+escape(str(ev.get('nurse','')))+'</td><td>'
                   +escape(str(ev.get('job','')))+'</td><td>'+escape(str(ev.get('detail','')))
                   +'</td></tr>' for i,ev in enumerate(events))
    html='''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>__TITLE__</title>
<style>body{background:#f4f7fa;color:#17283b;font:15px system-ui,Arial;margin:0;padding:0}
main{max-width:1350px;margin:auto;padding:28px}h1{margin:0 0 6px;font-size:28px}
header p{color:#57708b;margin-top:0}.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:10px;margin:24px 0}
.metric,.panel{background:white;border:1px solid #dfE5ec;border-radius:12px;padding:16px}
.metric .key{font-size:12px;color:#60758d}.metric .value{font-weight:700;font-size:22px;margin-top:6px}
.panel{margin-top:14px}input{padding:10px;border:1px solid #c7d4e0;border-radius:6px;min-width:260px}
table{width:100%;border-collapse:collapse;font-size:12px}td,th{padding:7px;border-bottom:1px solid #e7ebf0;text-align:left}
.scroller{height:330px;overflow:auto}tr:nth-child(even){background:#f7f9fc}.note{font-size:13px;color:#64748b}</style></head><body>
<main><header><h1>__TITLE__</h1><p>Original SHHCSRP data · SimPy execution · Gymnasium dispatch · Policy: <b>__POLICY__</b></p></header>
<div class="metrics">__CARDS__</div><section class="panel">__MAP__</section>
<section class="panel">__TIME__</section><section class="panel"><h2>Event trace</h2>
<p class="note">Search by event type, nurse, job or note. Time is simulated minutes, not wall-clock time.</p>
<input type="search" id="filter" placeholder="Filter events..." oninput="filterEvents()">
<div class="scroller"><table id="events"><thead><tr><th>#</th><th>Time</th><th>Event</th><th>Nurse</th><th>Job</th><th>Detail</th></tr></thead>
<tbody>__ROWS__</tbody></table></div></section>
<p class="note">Self-contained HTML with embedded Plotly. Travel geometry retains the legacy distance approximation. Service durations are sampled.
This is a demonstrator, not a clinically validated dispatch system.</p></main>
<script>function filterEvents(){let q=document.getElementById('filter').value.toLowerCase();for(let tr of document.querySelectorAll('#events tbody tr'))tr.style.display=tr.innerText.toLowerCase().includes(q)?'':'none';}</script></body></html>'''
    for key,val in {'__TITLE__':escape(title),'__POLICY__':escape(policy),
                    '__CARDS__':card_html,'__MAP__':chart1,'__TIME__':chart2,'__ROWS__':rows}.items():
        html=html.replace(key,val)
    output.write_text(html,encoding='utf-8')
    return output
