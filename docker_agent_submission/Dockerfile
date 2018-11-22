FROM python:3.6

ADD ./docker_agent_submission /agent

# @TODO to be replaced with `pip install pommerman`
ADD . /pommerman
RUN cd /pommerman && pip install .
# end @TODO

EXPOSE 10080

ENV NAME ES_Agent

# Run app.py when the container launches
WORKDIR /agent
ENTRYPOINT ["python"]
CMD ["run.py"]
