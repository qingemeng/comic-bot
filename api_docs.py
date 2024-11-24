api_doc = """
openapi: 3.0.0

info:
  version: 1.0.0
  title: xkcd
  description: 'A webcomic of romance, sarcasm, math, and language.'

servers:
  - url: https://xkcd.com/
    description: Official xkcd JSON interface

paths:
  # Retrieve the current comic
  /info.0.json:
    get:
      # A list of tags to logical group operations by resources and any other
      # qualifier.
      tags:
        - comic
      description: Returns comic based on ID
      summary: Find latest comic
      # Unique identifier for the operation, tools and libraries may use the
      # operationId to uniquely identify an operation.
      operationId: getComic
      responses:
        '200':
          description: Successfully returned a comic
          content:
            application/json:
              schema:
                # Relative reference to prevent duplicate schema definition.
                $ref: '#/components/schemas/Comic'
  # Retrieve a comic by ID
  /{id}/info.0.json:
    get:
      tags:
        - comic
      description: Returns comic based on ID
      summary: Find comic by ID
      operationId: getComicById
      parameters:
        - name: id
          in: path
          required: true
          schema:
            type: integer
      responses:
        '200':
          description: Successfully returned a commmic
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Comic'

components:
  schemas:
    Comic:
      type: object
      properties:
        month:
          type: string
        num:
          type: integer
        link:
          type: string
        year:
          type: string
        news:
          type: string
        safe_title:
          type: string
        transcript:
          type: string
        alt:
          type: string
        img:
          type: string
        title:
          type: string
        day:
          type: string
          """
