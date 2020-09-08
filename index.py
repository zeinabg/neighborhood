from flask import Flask, flash, g, make_response, redirect, render_template, request, session, url_for
from flask_wtf import FlaskForm
from wtforms import SelectMultipleField, StringField, SubmitField
from wtforms.validators import DataRequired, Length

from datetime import date
import enum
import os
import base64
from io import BytesIO

import numpy as np
import pandas as pd
import geopandas as geopd
from matplotlib import pyplot as plt
import matplotlib
import plotly.graph_objects as go


base_dir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY')
# app.secret_key = os.urandom(24)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(
    base_dir, 'repo', 'data.sqlite')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False


@app.before_first_request
def before_first_request_func():
  session['zipcodes'] = session.get('zipcodes', [])
  session['scores'] = session.get('scores', [])


@app.before_request
def before_request_func():
  geos = ['city', 'county_fips', 'county_name', 'geoid', 'zip']
  meta = pd.read_csv('repo/hpi_metadata.csv').set_index('variable')
  g.score_descriptions = meta.drop(geos + ['quintiles', 'quartiles', 'hpi_top25pct', 'urbantype', 'hpi2score', 'pop2010', 'pct2010gq'])['shorttitle'].to_dict()
  scores = list(g.score_descriptions.keys())
  pctile_scores = [s + '_pctile' for s in scores]
  hpi = pd.read_csv(
      'repo/hpi.csv', 
      dtype={'city': str, 'county_fips': str, 'county_name': str, 'geoid': str, 'zip': str}
  )
  g.hpi = np.round(hpi.groupby('zip')[pctile_scores].mean().rename(columns={s + '_pctile': s for s in scores})).dropna().astype(int) # todo: regenerate percentiles from weighted average on raw scores
  g.zipcode_geos = hpi[geos].drop_duplicates('zip').set_index('zip')
  g.shape = geopd.read_file('repo/california_zcta/californiazcta.shp').merge(g.hpi.reset_index(), left_on='ZCTA5CE10', right_on='zip')


@app.route('/')
def radar():
  return display_radar()


def display_radar():
  form_zip = ZipcodeForm()
  form_score = ScoreForm()
  form_score.scores.choices = [(score, desc) for score, desc in g.score_descriptions.items()]
  form_score.scores.data = session.get('scores')
  geo_values, score_values = retrieve_customized_data()
  radar = plot_radar(score_values)
  return render_template(
      'radar.html',
      geo_values=geo_values,
      score_values=score_values,
      radar=radar,
      form_zip=form_zip,
      form_score=form_score,
  )


@app.route('/zip', methods=['POST'])
def radar_zip():
  form_zip = ZipcodeForm()
  if form_zip.validate_on_submit():
    new_zipcode = form_zip.zipcode.data
    if not verify_zipcode(new_zipcode):
      flash('Zipcode is not valid.')
    elif new_zipcode in session.get('zipcodes'):
      flash('Zipcode already included.')
    else:
      flash('%s is added.' % new_zipcode)
      session['zipcodes'].append(new_zipcode)
    return redirect(url_for('radar'))
  return display_radar()


@app.route('/score', methods=['POST'])
def radar_score():
  form_score = ScoreForm()
  form_score.scores.choices = [(score, desc) for score, desc in g.score_descriptions.items()]
  if form_score.validate_on_submit():
    session['scores'] = form_score.scores.data
    return redirect(url_for('radar'))
  return display_radar()


@app.route('/map', methods=['GET', 'POST'])
def chmap():
  form_score = ScoreForm()
  form_score.scores.choices = [(score, desc) for score, desc in g.score_descriptions.items()]
  if form_score.validate_on_submit():
    session['scores'] = form_score.scores.data
    return redirect(url_for('chmap'))

  ch_map = plot_ch_map()
  form_score.scores.data = session.get('scores')
  return render_template(
      'chmap.html',
      ch_map=ch_map,
      form_score=form_score,
  )


class ZipcodeForm(FlaskForm):
  zipcode = StringField('Enter a zipcode in California', validators=[DataRequired()])
  submit = SubmitField('Search')


class ScoreForm(FlaskForm):
  scores = SelectMultipleField('Choose scores to show')
  submit = SubmitField('Submit')


def verify_zipcode(zipcode):
  return zipcode in g.get('hpi').index



def retrieve_customized_data():
  zipcodes = session.get('zipcodes')
  scores = session.get('scores')
  if not zipcodes:
    return pd.DataFrame(), pd.DataFrame()
  return g.get('zipcode_geos').loc[zipcodes], g.get('hpi').loc[zipcodes, scores].rename(columns=g.score_descriptions)


def plot_radar(score_values):
  scores = score_values.columns.tolist()
  fig = go.Figure()
  for zipcode, values in score_values.iterrows():
    fig.add_trace(go.Scatterpolar(
          r=values.tolist(),
          theta=scores,
          fill='toself',
          name=zipcode,
    ))
  fig.update_layout(
    polar=dict(
      radialaxis=dict(
        visible=True,
        range=[0, 100]
      )),
    showlegend=True
  )
  return fig


def plot_ch_map():
  matplotlib.use('Agg')
  scores = session.get('scores')
  if not scores:
    return ""
  col = int(np.ceil(np.sqrt(len(scores))))
  row = int(np.ceil(len(scores) / col))
  fig, axes = plt.subplots(row, col, figsize=(col * 3, row * 3))
  if row == 1 and col == 1:
    axes = np.array([axes])
  for ax, score in zip(axes.reshape(-1), scores):
    g.shape.plot(column=score, legend=True, ax=ax)
    ax.set_title(g.score_descriptions[score])
  fig.tight_layout()
  buf = BytesIO()
  fig.savefig(buf, format="png")
  data = base64.b64encode(buf.getbuffer()).decode("ascii")
  return "<img src='data:image/png;base64,%s'/>" % data


def setup_geofile():
  shp = geopd.read_file('repo/cb_2018_us_zcta510_500k/cb_2018_us_zcta510_500k.shp', dtype=str)
  ca_zips = pd.read_csv('repo/hpi.csv', dtype=str)
  ca_shp = shp.loc[shp.ZCTA5CE10.isin(ca_zips.zip)]
  ca_shp.to_file('repo/california_zcta')