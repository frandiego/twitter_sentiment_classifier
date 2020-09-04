.PHONY: help data



help:
	@echo "Please use 'make <target> [params]' where <target> is one of"
	@echo "  data                   Download sentiment140 dataset"
	@echo " "


raw-data:
	@python etl_download.py

sample-sentiment-analysis:
	@python etl_sentiment.py

etl: raw-data sample-sentiment-analysis

