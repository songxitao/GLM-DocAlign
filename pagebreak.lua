function RawBlock(el)
  if (el.format == "tex" or el.format == "latex") and (el.text:match("\\newpage") or el.text:match("\\clearpage")) then
    return pandoc.RawBlock('openxml', '<w:p><w:r><w:br w:type="page"/></w:r></w:p>')
  end
end

function Para(el)
  if #el.content == 1 and el.content[1].text == "\\newpage" then
    return pandoc.RawBlock('openxml', '<w:p><w:r><w:br w:type="page"/></w:r></w:p>')
  end
end