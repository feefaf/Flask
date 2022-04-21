var form = document.forms.para;
document.getElementByClassName("form-select form-select-sm").addEventListener("change", function () {
  form.submit();
});