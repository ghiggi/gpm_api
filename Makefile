# Pass arguments to make: https://stackoverflow.com/a/6273809

changelog:
	@echo "Updating CHANGELOG.md..."

	@# Generate CHANGELOG.temp
	@loghub ghiggi/pycolorbar

	@# Put version in CHANGELOG.temp
	@sed -i 's/<RELEASE_VERSION>/$(filter-out $@,$(MAKECMDGOALS))/g' CHANGELOG.temp

	@# Remove "# Changelog" header from CHANGELOG.md
	@sed -i '/# Changelog/d' CHANGELOG.md

	@# Append CHANGELOG.temp to the top of CHANGELOG.md
	@cat CHANGELOG.md >> CHANGELOG.temp
	@mv CHANGELOG.temp CHANGELOG.md

	@# Add "# Changelog" header to CHANGELOG.md
	@sed -i '1s/^/# Changelog\n\n/' CHANGELOG.md

	@echo "Done."

%:
	@:
